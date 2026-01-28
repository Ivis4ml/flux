"""
Communication layer for Flux coordinator.

Provides ZeroMQ-based communication for coordinator-worker interaction.
Supports:
- Request-reply pattern for synchronous operations (ROUTER/DEALER)
- Pub-sub pattern for broadcasting weight updates (PUB/SUB)

Note: HTTP fallback is not currently implemented. When ZeroMQ is unavailable,
the CommunicationManager will operate without a backend (logging warnings on
send/broadcast attempts).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

import torch

try:
    import zmq
    import zmq.asyncio

    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

try:
    import grpc
    from concurrent import futures

    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False


logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages in the communication protocol."""

    # Control messages
    HEARTBEAT = "heartbeat"
    REGISTER = "register"
    DEREGISTER = "deregister"

    # Training messages
    WEIGHT_UPDATE = "weight_update"
    GRADIENT = "gradient"
    BATCH_REQUEST = "batch_request"
    BATCH_RESPONSE = "batch_response"

    # Rollout messages
    ROLLOUT_REQUEST = "rollout_request"
    ROLLOUT_RESPONSE = "rollout_response"
    ABORT_REQUEST = "abort_request"

    # Status messages
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"


def _serialize_payload(payload: Any) -> tuple[str, str | None]:
    """Serialize payload safely.

    For simple types, uses JSON. For tensors/complex types,
    uses torch.save with base64 encoding.

    Returns:
        Tuple of (serialization_type, serialized_data)
    """
    # Handle None
    if payload is None:
        return ("json", json.dumps(None))

    # Handle dict with potential tensors
    if isinstance(payload, dict):
        has_tensor = any(isinstance(v, torch.Tensor) for v in payload.values())
        if has_tensor:
            # Use torch serialization for tensor-containing dicts
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return ("torch", encoded)
        else:
            # Try JSON for simple dicts
            try:
                return ("json", json.dumps(payload))
            except (TypeError, ValueError):
                # Fall back to torch for non-JSON-serializable
                buffer = io.BytesIO()
                torch.save(payload, buffer)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                return ("torch", encoded)

    # Handle torch tensors directly
    if isinstance(payload, torch.Tensor):
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return ("torch", encoded)

    # Try JSON for everything else
    try:
        return ("json", json.dumps(payload))
    except (TypeError, ValueError):
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return ("torch", encoded)


def _deserialize_payload(ser_type: str, data: str) -> Any:
    """Deserialize payload.

    Args:
        ser_type: Serialization type ("json" or "torch")
        data: Serialized data string

    Returns:
        Deserialized payload
    """
    if ser_type == "json":
        return json.loads(data)
    elif ser_type == "torch":
        decoded = base64.b64decode(data.encode("ascii"))
        buffer = io.BytesIO(decoded)
        return torch.load(buffer, weights_only=True)
    else:
        raise ValueError(f"Unknown serialization type: {ser_type}")


@dataclass
class Message:
    """A message in the communication protocol."""

    msg_type: MessageType
    payload: Any
    sender_id: str = ""
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""

    def serialize(self) -> bytes:
        """Serialize message to bytes using safe JSON/torch serialization."""
        ser_type, payload_data = _serialize_payload(self.payload)

        data = {
            "type": self.msg_type.value,
            "ser_type": ser_type,
            "payload": payload_data,
            "sender": self.sender_id,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        """Deserialize message from bytes."""
        d = json.loads(data.decode("utf-8"))
        payload = _deserialize_payload(d["ser_type"], d["payload"])
        return cls(
            msg_type=MessageType(d["type"]),
            payload=payload,
            sender_id=d["sender"],
            timestamp=d["timestamp"],
            request_id=d.get("request_id", ""),
        )


class CommunicationBackend(ABC):
    """Abstract base class for communication backends."""

    @abstractmethod
    async def start(self) -> None:
        """Start the communication backend."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the communication backend."""
        pass

    @abstractmethod
    async def send(self, message: Message, target: str) -> None:
        """Send a message to a target."""
        pass

    @abstractmethod
    async def receive(self, timeout: float | None = None) -> Message | None:
        """Receive a message."""
        pass

    @abstractmethod
    async def broadcast(self, message: Message) -> None:
        """Broadcast a message to all connected clients."""
        pass


class ZMQBackend(CommunicationBackend):
    """ZeroMQ-based communication backend.

    Uses multiple socket patterns:
    - ROUTER/DEALER for request-reply
    - PUB/SUB for broadcasting
    - PUSH/PULL for task distribution

    Example:
        backend = ZMQBackend(
            router_addr="tcp://*:5555",
            pub_addr="tcp://*:5556",
        )
        await backend.start()

        # Send to specific worker
        await backend.send(message, worker_id)

        # Broadcast to all
        await backend.broadcast(weight_update_message)
    """

    def __init__(
        self,
        router_addr: str = "tcp://*:5555",
        pub_addr: str = "tcp://*:5556",
        identity: str = "coordinator",
    ) -> None:
        """Initialize ZMQ backend.

        Args:
            router_addr: Address for ROUTER socket (request-reply).
            pub_addr: Address for PUB socket (broadcast).
            identity: Identity for this endpoint.
        """
        if not HAS_ZMQ:
            raise RuntimeError("ZeroMQ not installed. Install with: pip install pyzmq")

        self.router_addr = router_addr
        self.pub_addr = pub_addr
        self.identity = identity

        self._context: zmq.asyncio.Context | None = None
        self._router: zmq.asyncio.Socket | None = None
        self._publisher: zmq.asyncio.Socket | None = None
        self._running = False

        # Connected workers
        self._workers: dict[str, float] = {}  # worker_id -> last_seen

    async def start(self) -> None:
        """Start the ZMQ backend."""
        if self._running:
            return

        self._context = zmq.asyncio.Context()

        # ROUTER for request-reply with workers
        self._router = self._context.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.IDENTITY, self.identity.encode())
        self._router.bind(self.router_addr)

        # PUB for broadcasting updates
        self._publisher = self._context.socket(zmq.PUB)
        self._publisher.bind(self.pub_addr)

        self._running = True
        logger.info(
            f"ZMQ backend started: ROUTER={self.router_addr}, PUB={self.pub_addr}"
        )

    async def stop(self) -> None:
        """Stop the ZMQ backend."""
        if not self._running:
            return

        self._running = False

        if self._router:
            self._router.close()
            self._router = None

        if self._publisher:
            self._publisher.close()
            self._publisher = None

        if self._context:
            self._context.term()
            self._context = None

        logger.info("ZMQ backend stopped")

    async def send(self, message: Message, target: str) -> None:
        """Send a message to a specific target."""
        if not self._router:
            raise RuntimeError("Backend not started")

        # ROUTER sends: [identity, empty, data]
        frames = [
            target.encode(),
            b"",
            message.serialize(),
        ]
        await self._router.send_multipart(frames)

    async def receive(self, timeout: float | None = None) -> Message | None:
        """Receive a message from workers."""
        if not self._router:
            raise RuntimeError("Backend not started")

        try:
            if timeout:
                self._router.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
            else:
                self._router.setsockopt(zmq.RCVTIMEO, -1)

            # ROUTER receives: [identity, empty, data]
            frames = await self._router.recv_multipart()
            if len(frames) >= 3:
                sender_id = frames[0].decode()
                data = frames[2]
                message = Message.deserialize(data)
                message.sender_id = sender_id

                # Update worker last-seen
                self._workers[sender_id] = time.time()

                return message
            return None

        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None

    async def broadcast(self, message: Message) -> None:
        """Broadcast a message to all subscribers."""
        if not self._publisher:
            raise RuntimeError("Backend not started")

        # PUB sends: [topic, data]
        topic = message.msg_type.value.encode()
        await self._publisher.send_multipart([topic, message.serialize()])

    def get_connected_workers(self) -> list[str]:
        """Get list of connected worker IDs."""
        # Consider workers stale after 60 seconds
        cutoff = time.time() - 60
        return [w for w, t in self._workers.items() if t > cutoff]


class ZMQWorkerClient:
    """ZMQ client for workers connecting to coordinator.

    Example:
        client = ZMQWorkerClient(
            coordinator_addr="tcp://localhost:5555",
            sub_addr="tcp://localhost:5556",
            worker_id="worker-0",
        )
        await client.start()

        # Send request to coordinator
        response = await client.request(message)

        # Subscribe to weight updates
        async for msg in client.subscribe(MessageType.WEIGHT_UPDATE):
            update_weights(msg.payload)
    """

    def __init__(
        self,
        coordinator_addr: str = "tcp://localhost:5555",
        sub_addr: str = "tcp://localhost:5556",
        worker_id: str = "worker",
    ) -> None:
        """Initialize worker client.

        Args:
            coordinator_addr: Coordinator ROUTER address.
            sub_addr: Coordinator PUB address for subscriptions.
            worker_id: Unique worker identifier.
        """
        if not HAS_ZMQ:
            raise RuntimeError("ZeroMQ not installed")

        self.coordinator_addr = coordinator_addr
        self.sub_addr = sub_addr
        self.worker_id = worker_id

        self._context: zmq.asyncio.Context | None = None
        self._dealer: zmq.asyncio.Socket | None = None
        self._subscriber: zmq.asyncio.Socket | None = None
        self._running = False

    async def start(self) -> None:
        """Start the client."""
        if self._running:
            return

        self._context = zmq.asyncio.Context()

        # DEALER for request-reply with coordinator
        self._dealer = self._context.socket(zmq.DEALER)
        self._dealer.setsockopt(zmq.IDENTITY, self.worker_id.encode())
        self._dealer.connect(self.coordinator_addr)

        # SUB for receiving broadcasts
        self._subscriber = self._context.socket(zmq.SUB)
        self._subscriber.connect(self.sub_addr)

        self._running = True
        logger.info(f"Worker client started: {self.worker_id}")

    async def stop(self) -> None:
        """Stop the client."""
        if not self._running:
            return

        self._running = False

        if self._dealer:
            self._dealer.close()
        if self._subscriber:
            self._subscriber.close()
        if self._context:
            self._context.term()

    async def send(self, message: Message) -> None:
        """Send a message to coordinator."""
        if not self._dealer:
            raise RuntimeError("Client not started")

        message.sender_id = self.worker_id
        frames = [b"", message.serialize()]
        await self._dealer.send_multipart(frames)

    async def request(
        self, message: Message, timeout: float = 30.0
    ) -> Message | None:
        """Send request and wait for response."""
        if not self._dealer:
            raise RuntimeError("Client not started")

        await self.send(message)

        try:
            self._dealer.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
            frames = await self._dealer.recv_multipart()
            if len(frames) >= 2:
                return Message.deserialize(frames[1])
            return None
        except zmq.Again:
            logger.warning("Request timed out")
            return None

    def subscribe(self, *topics: MessageType) -> None:
        """Subscribe to message types."""
        if not self._subscriber:
            raise RuntimeError("Client not started")

        for topic in topics:
            self._subscriber.setsockopt(zmq.SUBSCRIBE, topic.value.encode())

    async def receive_broadcast(
        self, timeout: float | None = None
    ) -> Message | None:
        """Receive a broadcast message."""
        if not self._subscriber:
            raise RuntimeError("Client not started")

        try:
            if timeout:
                self._subscriber.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
            else:
                self._subscriber.setsockopt(zmq.RCVTIMEO, -1)

            frames = await self._subscriber.recv_multipart()
            if len(frames) >= 2:
                return Message.deserialize(frames[1])
            return None
        except zmq.Again:
            return None


class CommunicationManager:
    """Manages communication for the coordinator.

    Provides a unified interface for ZMQ communication. When ZMQ is unavailable,
    operates without a backend (send/broadcast will log warnings).

    Example:
        manager = CommunicationManager(
            use_zmq=True,
            zmq_router_addr="tcp://*:5555",
        )
        await manager.start()

        # Broadcast weight update
        await manager.broadcast_weights(new_weights, version=10)

        # Receive messages (with optional timeout)
        message = await manager.receive(timeout=1.0)
        if message:
            # Process message and send response
            response = Message(msg_type=MessageType.STATUS_RESPONSE, payload={...})
            await manager.send(response, target=message.sender_id)

        # Or run the server loop to handle messages via registered handlers
        manager.register_handler(MessageType.BATCH_REQUEST, handle_batch_request)
        await manager.run_server()
    """

    def __init__(
        self,
        use_zmq: bool = True,
        zmq_router_addr: str = "tcp://*:5555",
        zmq_pub_addr: str = "tcp://*:5556",
        identity: str = "coordinator",
    ) -> None:
        """Initialize communication manager.

        Args:
            use_zmq: Whether to use ZMQ. If False or ZMQ unavailable, operates
                without a backend (send/broadcast will log warnings).
            zmq_router_addr: ZMQ ROUTER address.
            zmq_pub_addr: ZMQ PUB address.
            identity: Identity for this endpoint.
        """
        self.use_zmq = use_zmq and HAS_ZMQ
        self._backend: CommunicationBackend | None = None
        self._running = False

        # Request handlers
        self._handlers: dict[MessageType, Callable] = {}

        # Stats
        self._messages_sent = 0
        self._messages_received = 0
        self._broadcasts_sent = 0

        if self.use_zmq:
            self._backend = ZMQBackend(
                router_addr=zmq_router_addr,
                pub_addr=zmq_pub_addr,
                identity=identity,
            )

    async def start(self) -> None:
        """Start the communication manager."""
        if self._running:
            return

        if self._backend:
            await self._backend.start()

        self._running = True
        logger.info(f"Communication manager started (ZMQ: {self.use_zmq})")

    async def stop(self) -> None:
        """Stop the communication manager."""
        if not self._running:
            return

        self._running = False

        if self._backend:
            await self._backend.stop()

        logger.info("Communication manager stopped")

    def register_handler(
        self, msg_type: MessageType, handler: Callable
    ) -> None:
        """Register a message handler.

        Args:
            msg_type: Message type to handle.
            handler: Async function to handle messages of this type.
        """
        self._handlers[msg_type] = handler

    async def send(self, message: Message, target: str) -> None:
        """Send a message to a specific target."""
        if self._backend:
            await self._backend.send(message, target)
            self._messages_sent += 1
        else:
            logger.warning("No backend available for send")

    async def broadcast(self, message: Message) -> None:
        """Broadcast a message to all connected clients."""
        if self._backend:
            await self._backend.broadcast(message)
            self._broadcasts_sent += 1
        else:
            logger.warning("No backend available for broadcast")

    async def broadcast_weights(
        self, weights: dict[str, Any], version: int
    ) -> None:
        """Broadcast weight update to all workers."""
        message = Message(
            msg_type=MessageType.WEIGHT_UPDATE,
            payload={"weights": weights, "version": version},
        )
        await self.broadcast(message)

    async def receive(self, timeout: float | None = None) -> Message | None:
        """Receive a message."""
        if self._backend:
            msg = await self._backend.receive(timeout)
            if msg:
                self._messages_received += 1
            return msg
        return None

    async def run_server(self) -> None:
        """Run the message server loop."""
        while self._running:
            try:
                message = await self.receive(timeout=1.0)
                if message and message.msg_type in self._handlers:
                    handler = self._handlers[message.msg_type]
                    try:
                        response = await handler(message)
                        if response and message.sender_id:
                            await self.send(response, message.sender_id)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")
            except Exception as e:
                logger.error(f"Server loop error: {e}")
                await asyncio.sleep(0.1)

    def get_stats(self) -> dict[str, Any]:
        """Get communication statistics."""
        workers = []
        if isinstance(self._backend, ZMQBackend):
            workers = self._backend.get_connected_workers()

        return {
            "use_zmq": self.use_zmq,
            "running": self._running,
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "broadcasts_sent": self._broadcasts_sent,
            "connected_workers": workers,
        }
