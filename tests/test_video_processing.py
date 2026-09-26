import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from evovideo_skill.video_processing import VideoProcessor


class VideoProcessingTest(unittest.TestCase):
    def test_processes_local_file_url_and_extracts_frame_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                4,
                (64, 64),
            )
            for idx in range(8):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                frame[:, :] = (20, 20, 220) if idx < 4 else (40, 180, 180)
                writer.write(frame)
            writer.release()

            processor = VideoProcessor(output_dir=Path(tmpdir) / "out", sample_count=4)
            result = processor.process(video_path.as_uri(), "local-test", "a red coat walks", {"temporal_steps": ["walks"]})

            self.assertTrue(Path(result.local_video_path).exists())
            self.assertEqual(len(result.sampled_frame_paths), 4)
            self.assertEqual(len(result.frames), 4)
            self.assertIn("frame_path", result.frames[0])
            self.assertIn("color_presence", result.frames[0])

    def test_reuses_local_video_already_in_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir()
            video_path = output_dir / "wan-original.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                4,
                (32, 32),
            )
            for _ in range(4):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()

            result = VideoProcessor(output_dir=output_dir, sample_count=2).process(
                video_path.as_uri(),
                "renamed-copy",
                "A static scene",
            )

            self.assertEqual(Path(result.local_video_path), video_path.resolve())
            self.assertFalse((output_dir / "renamed-copy.mp4").exists())
            self.assertEqual(len(result.sampled_frame_paths), 2)


if __name__ == "__main__":
    unittest.main()
