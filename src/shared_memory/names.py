from dataclasses import dataclass


@dataclass(frozen=True)
class SharedMemoryNames:
    prefix: str

    @property
    def camera_prefix(self) -> str:
        """Return the shared-memory prefix used for camera arrays."""
        return f"{self.prefix}_camera"

    @property
    def lidar_prefix(self) -> str:
        """Return the shared-memory prefix used for LiDAR arrays."""
        return f"{self.prefix}_lidar"

    @property
    def frame_buffer(self) -> str:
        """Return the shared-memory name used for frame snapshots."""
        return f"{self.prefix}_frame"

    @property
    def objects_buffer(self) -> str:
        """Return the shared-memory name used for object predictions."""
        return f"{self.prefix}_objects"

    @property
    def lanes_buffer(self) -> str:
        """Return the shared-memory name used for lane predictions."""
        return f"{self.prefix}_lanes"
