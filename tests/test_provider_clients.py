import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evovideo_skill.provider_clients import (
    AliyunWanxClient,
    HailuoClient,
    KlingClient,
    ProviderConfig,
    provider_from_env,
)


class FakeHttp:
    def __init__(self):
        self.requests = []

    def request_json(self, method, url, headers, payload=None, query=None):
        self.requests.append((method, url, payload, query))
        if "minimaxi" in url and method == "POST":
            return {"task_id": "mini-task", "base_resp": {"status_code": 0}}
        if "query/video_generation" in url:
            return {"task_id": "mini-task", "status": "Success", "file_id": "123"}
        if "files/retrieve" in url:
            return {"file": {"download_url": "https://example.com/minimax.mp4"}}
        if "video-synthesis" in url:
            return {"output": {"task_id": "ali-task"}}
        if "/api/v1/tasks/" in url:
            return {"output": {"task_status": "SUCCEEDED", "video_url": "https://example.com/aliyun.mp4"}}
        if "text2video" in url and method == "POST":
            return {"data": {"task_id": "kling-task"}}
        if "text2video" in url and method == "GET":
            return {"data": {"task_status": "succeed", "task_result": {"videos": [{"url": "https://example.com/kling.mp4"}]}}}
        raise AssertionError(f"unexpected request: {method} {url}")


def config(provider):
    return ProviderConfig(
        provider=provider,
        model_t2v="t2v",
        model_i2v="i2v",
        model_edit="edit",
        duration=6,
        resolution="768P",
        poll_interval_seconds=0,
        timeout_seconds=10,
    )


class ProviderClientTest(unittest.TestCase):
    def test_hailuo_maps_to_uniform_response(self):
        client = HailuoClient(config("hailuo"), api_key="key", http=FakeHttp())
        result = client.create_video("/v1/videos/generations", {"prompt": "robot walks", "duration_seconds": 6, "mode": "generation"})
        self.assertEqual(result["provider"], "hailuo")
        self.assertEqual(result["video_url"], "https://example.com/minimax.mp4")
        self.assertTrue(result["frames"])

    def test_aliyun_maps_to_uniform_response(self):
        http = FakeHttp()
        client = AliyunWanxClient(config("aliyun-wanx"), api_key="key", http=http)
        result = client.create_video("/v1/videos/generations", {"prompt": "robot walks", "duration_seconds": 6, "mode": "generation"})
        post_payload = next(payload for method, url, payload, _ in http.requests if method == "POST" and "video-synthesis" in url)
        self.assertEqual(result["provider"], "aliyun-wanx")
        self.assertEqual(result["video_url"], "https://example.com/aliyun.mp4")
        self.assertNotIn("duration", post_payload["parameters"])
        self.assertEqual(post_payload["parameters"]["size"], "1280*720")

    def test_aliyun_size_mapping_uses_supported_dashscope_sizes(self):
        self.assertEqual(AliyunWanxClient._dashscope_size("768P"), "1280*720")
        self.assertEqual(AliyunWanxClient._dashscope_size("1088*832"), "1088*832")
        self.assertEqual(AliyunWanxClient._dashscope_size("1280*768"), "1280*720")

    def test_aliyun_duration_is_opt_in(self):
        http = FakeHttp()
        with patch.dict("os.environ", {"EVOVIDEO_WANX_ENABLE_DURATION": "1"}):
            client = AliyunWanxClient(config("aliyun-wanx"), api_key="key", http=http)
            client.create_video("/v1/videos/generations", {"prompt": "robot walks", "duration_seconds": 6, "mode": "generation"})
        post_payload = next(payload for method, url, payload, _ in http.requests if method == "POST" and "video-synthesis" in url)
        self.assertEqual(post_payload["parameters"]["duration"], 6)

    def test_aliyun_i2v_without_reference_falls_back_to_t2v(self):
        http = FakeHttp()
        client = AliyunWanxClient(config("aliyun-wanx"), api_key="key", http=http)
        result = client.create_video("/v1/videos/generations", {"prompt": "robot walks", "duration_seconds": 6, "mode": "image_to_video"})
        post_payload = next(payload for method, url, payload, _ in http.requests if method == "POST" and "video-synthesis" in url)
        self.assertEqual(result["provider"], "aliyun-wanx")
        self.assertEqual(post_payload["model"], "t2v")
        self.assertNotIn("img_url", post_payload["input"])

    def test_kling_maps_to_uniform_response(self):
        client = KlingClient(config("kling"), access_key="ak", secret_key="sk", http=FakeHttp())
        result = client.create_video("/v1/videos/generations", {"prompt": "robot walks", "duration_seconds": 6, "mode": "generation"})
        self.assertEqual(result["provider"], "kling")
        self.assertEqual(result["video_url"], "https://example.com/kling.mp4")

    def test_auto_prefers_hailuo_then_aliyun_then_kling(self):
        with patch.dict("os.environ", {"MINIMAX_API_KEY": "key"}, clear=True):
            self.assertIsInstance(provider_from_env("auto", type("Args", (), {})()), HailuoClient)
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "key"}, clear=True):
            self.assertIsInstance(provider_from_env("auto", type("Args", (), {})()), AliyunWanxClient)
        with patch.dict("os.environ", {"KLING_ACCESS_KEY": "ak", "KLING_SECRET_KEY": "sk"}, clear=True):
            self.assertIsInstance(provider_from_env("auto", type("Args", (), {})()), KlingClient)


if __name__ == "__main__":
    unittest.main()
