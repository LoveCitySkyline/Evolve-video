from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from evovideo_skill.api_tools import VideoApiError
from evovideo_skill.video_processing import VideoProcessor


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass
class ProviderConfig:
    provider: str
    model_t2v: str
    model_i2v: str
    model_edit: str
    duration: int
    resolution: str
    poll_interval_seconds: float
    timeout_seconds: int
    reference_image_url: str | None = None
    process_video: bool = True
    video_output_dir: str = "outputs/provider_videos"
    sample_frames: int = 6


class JsonHttpClient:
    def __init__(self, timeout_seconds: int = 120):
        self.timeout_seconds = timeout_seconds

    def request_json(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VideoApiError(f"{method} {url} failed: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise VideoApiError(f"{method} {url} failed: {exc}") from exc
        return json.loads(body) if body else {}


class ProviderVideoClient:
    provider_name = "provider"

    def __init__(self, config: ProviderConfig, http: JsonHttpClient | None = None):
        self.config = config
        self.http = http or JsonHttpClient(timeout_seconds=config.timeout_seconds)
        self.video_processor = VideoProcessor(
            output_dir=config.video_output_dir,
            sample_count=config.sample_frames,
            timeout_seconds=config.timeout_seconds,
        )

    def create_video(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload.get("mode")
        if route.endswith("/edits") or mode in {"global_editing", "region_editing"}:
            return self.create_edit(payload)
        if mode == "image_to_video":
            return self.create_i2v(payload)
        return self.create_t2v(payload)

    def create_t2v(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def create_i2v(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create_t2v(payload)

    def create_edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create_t2v(payload)

    def _frames_from_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        plan = payload.get("plan") or {}
        intent = plan.get("intent") or {}
        steps = plan.get("temporal_steps") or ["generated video"]
        duration = int(payload.get("duration_seconds") or self.config.duration or 6)
        frames = []
        for idx in range(max(4, duration)):
            frames.append(
                {
                    "index": idx,
                    "subject": intent.get("subject", "subject"),
                    "identity": "unknown_without_frame_sampler",
                    "clothing_color": intent.get("clothing_color"),
                    "action": steps[min(idx * len(steps) // max(1, duration), len(steps) - 1)],
                    "style": intent.get("style", "default"),
                    "background_changed": False,
                    "target_edit_success": True,
                }
            )
        return frames

    def _uniform_response(
        self,
        payload: dict[str, Any],
        video_id: str,
        video_url: str | None,
        raw: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        frames = self._frames_from_payload(payload)
        combined_metadata = {"raw": raw, **(metadata or {})}
        combined_metadata.setdefault("original_prompt", payload.get("original_prompt") or payload.get("prompt"))
        combined_metadata.setdefault("generation_prompt", payload.get("prompt"))
        combined_metadata.setdefault("prompt_rewrite_reasons", payload.get("prompt_rewrite_reasons", []))
        if video_url and self.config.process_video:
            try:
                processed = self.video_processor.process(video_url, video_id, payload.get("prompt", ""), payload)
                frames = processed.frames or frames
                combined_metadata.update(
                    {
                        "local_video_path": processed.local_video_path,
                        "sampled_frame_paths": processed.sampled_frame_paths,
                        "real_video_processed": True,
                    }
                )
            except Exception as exc:
                combined_metadata.update(
                    {
                        "real_video_processed": False,
                        "video_processing_error": str(exc),
                    }
                )
        return {
            "provider": self.provider_name,
            "video_id": video_id,
            "video_url": video_url,
            "frames": frames,
            "metadata": combined_metadata,
        }


class HailuoClient(ProviderVideoClient):
    provider_name = "hailuo"

    def __init__(self, config: ProviderConfig, api_key: str | None = None, base_url: str | None = None, http: JsonHttpClient | None = None):
        super().__init__(config, http=http)
        self.api_key = api_key or env("MINIMAX_API_KEY")
        if not self.api_key:
            raise VideoApiError("MINIMAX_API_KEY is required for provider=hailuo")
        self.base_url = (base_url or env("MINIMAX_BASE_URL") or "https://api.minimaxi.com").rstrip("/")

    def create_t2v(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._create_video_task(
            {
                "model": self.config.model_t2v,
                "prompt": payload["prompt"],
                "duration": self.config.duration,
                "resolution": self.config.resolution,
                "prompt_optimizer": True,
            },
            payload,
        )

    def create_i2v(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_url = self.config.reference_image_url or env("EVOVIDEO_REFERENCE_IMAGE_URL")
        if not image_url:
            return self.create_t2v(payload)
        return self._create_video_task(
            {
                "model": self.config.model_i2v,
                "prompt": payload["prompt"],
                "first_frame_image": image_url,
                "duration": self.config.duration,
                "resolution": self.config.resolution,
                "prompt_optimizer": True,
            },
            payload,
        )

    def create_edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Hailuo's public video generation APIs expose T2V/I2V variants. For
        # agent-level workflow validation, map edits to an instruction prompt.
        edit_prompt = f"{payload['prompt']} Keep all non-target regions unchanged."
        return self._create_video_task(
            {
                "model": self.config.model_edit,
                "prompt": edit_prompt,
                "duration": self.config.duration,
                "resolution": self.config.resolution,
                "prompt_optimizer": True,
            },
            payload,
        )

    def _create_video_task(self, body: dict[str, Any], original_payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        created = self.http.request_json("POST", f"{self.base_url}/v1/video_generation", headers, body)
        base_resp = created.get("base_resp") or {}
        if base_resp.get("status_code") not in (None, 0):
            raise VideoApiError(
                "Hailuo request failed before task creation: "
                f"status_code={base_resp.get('status_code')}, "
                f"status_msg={base_resp.get('status_msg')}, "
                f"model={body.get('model')}. "
                "For Text-to-Video, use MiniMax-Hailuo-2.3, MiniMax-Hailuo-02, T2V-01-Director, or T2V-01. "
                "MiniMax-Hailuo-2.3-Fast is for Image-to-Video."
            )
        task_id = created.get("task_id")
        if not task_id:
            raise VideoApiError(f"Hailuo did not return task_id: {created}")
        status = self._wait_task(task_id)
        file_id = status.get("file_id")
        video_url = None
        if file_id:
            retrieved = self.http.request_json("GET", f"{self.base_url}/v1/files/retrieve", headers, query={"file_id": file_id})
            video_url = (retrieved.get("file") or {}).get("download_url")
        return self._uniform_response(
            original_payload,
            video_id=str(task_id),
            video_url=video_url,
            raw={"create": created, "status": status},
            metadata={"file_id": file_id, "model": body.get("model")},
        )

    def _wait_task(self, task_id: str) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        deadline = time.time() + self.config.timeout_seconds
        while time.time() < deadline:
            status = self.http.request_json("GET", f"{self.base_url}/v1/query/video_generation", headers, query={"task_id": task_id})
            state = status.get("status")
            if state == "Success":
                return status
            if state == "Fail":
                raise VideoApiError(f"Hailuo task failed: {status}")
            time.sleep(self.config.poll_interval_seconds)
        raise VideoApiError(f"Hailuo task timed out: {task_id}")


class AliyunWanxClient(ProviderVideoClient):
    provider_name = "aliyun-wanx"

    def __init__(self, config: ProviderConfig, api_key: str | None = None, base_url: str | None = None, http: JsonHttpClient | None = None):
        super().__init__(config, http=http)
        self.api_key = api_key or env("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise VideoApiError("DASHSCOPE_API_KEY is required for provider=aliyun-wanx")
        self.base_url = (base_url or env("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com").rstrip("/")

    def create_t2v(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._create_task(
            self.config.model_t2v,
            {"prompt": payload["prompt"]},
            payload,
        )

    def create_i2v(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_url = self.config.reference_image_url or env("EVOVIDEO_REFERENCE_IMAGE_URL")
        if not image_url:
            return self.create_t2v(payload)
        model = self.config.model_i2v
        input_payload = {"prompt": payload["prompt"]}
        input_payload["img_url"] = image_url
        return self._create_task(model, input_payload, payload)

    def create_edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._create_task(
            self.config.model_edit,
            {"prompt": f"{payload['prompt']} Keep non-target regions unchanged."},
            payload,
        )

    def _create_task(self, model: str, input_payload: dict[str, Any], original_payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        parameters = {
            "size": self._dashscope_size(self.config.resolution),
        }
        if env("EVOVIDEO_WANX_ENABLE_DURATION") == "1":
            parameters["duration"] = self.config.duration
        body = {
            "model": model,
            "input": input_payload,
            "parameters": parameters,
        }
        created = self.http.request_json("POST", f"{self.base_url}/api/v1/services/aigc/video-generation/video-synthesis", headers, body)
        task_id = ((created.get("output") or {}).get("task_id") or created.get("task_id") or created.get("request_id"))
        if not task_id:
            raise VideoApiError(f"Aliyun Wanx did not return task id: {created}")
        status = self._wait_task(task_id)
        output = status.get("output") or {}
        video_url = output.get("video_url") or output.get("url")
        return self._uniform_response(
            original_payload,
            video_id=str(task_id),
            video_url=video_url,
            raw={"create": created, "status": status},
            metadata={"model": model},
        )

    def _wait_task(self, task_id: str) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        deadline = time.time() + self.config.timeout_seconds
        while time.time() < deadline:
            status = self.http.request_json("GET", f"{self.base_url}/api/v1/tasks/{task_id}", headers)
            state = ((status.get("output") or {}).get("task_status") or status.get("task_status") or status.get("status"))
            if state in {"SUCCEEDED", "succeeded", "SUCCESS", "Success"}:
                return status
            if state in {"FAILED", "failed", "FAIL", "Fail"}:
                raise VideoApiError(f"Aliyun Wanx task failed: {status}")
            time.sleep(self.config.poll_interval_seconds)
        raise VideoApiError(f"Aliyun Wanx task timed out: {task_id}")

    @staticmethod
    def _dashscope_size(resolution: str) -> str:
        normalized = resolution.upper()
        table = {
            "480P": "832*480",
            "512P": "832*512",
            "720P": "1280*720",
            "768P": "1280*720",
            "1080P": "1280*720",
            "SQUARE": "960*960",
            "PORTRAIT": "720*1280",
            "LANDSCAPE": "1280*720",
        }
        supported = {"1280*720", "960*960", "720*1280", "1088*832", "832*1088", "832*480", "624*624", "480*832"}
        if normalized in table:
            return table[normalized]
        return resolution if resolution in supported else "1280*720"


class KlingClient(ProviderVideoClient):
    provider_name = "kling"

    def __init__(
        self,
        config: ProviderConfig,
        access_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        http: JsonHttpClient | None = None,
    ):
        super().__init__(config, http=http)
        self.access_key = access_key or env("KLING_ACCESS_KEY")
        self.secret_key = secret_key or env("KLING_SECRET_KEY")
        if not self.access_key or not self.secret_key:
            raise VideoApiError("KLING_ACCESS_KEY and KLING_SECRET_KEY are required for provider=kling")
        self.base_url = (base_url or env("KLING_BASE_URL") or "https://api-singapore.klingai.com").rstrip("/")

    def create_t2v(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._create_task(
            "/v1/videos/text2video",
            {
                "model_name": self.config.model_t2v,
                "prompt": payload["prompt"],
                "duration": str(self.config.duration),
                "mode": "std",
            },
            payload,
        )

    def create_i2v(self, payload: dict[str, Any]) -> dict[str, Any]:
        image_url = self.config.reference_image_url or env("EVOVIDEO_REFERENCE_IMAGE_URL")
        if not image_url:
            return self.create_t2v(payload)
        return self._create_task(
            "/v1/videos/image2video",
            {
                "model_name": self.config.model_i2v,
                "image": image_url,
                "prompt": payload["prompt"],
                "duration": str(self.config.duration),
                "mode": "std",
            },
            payload,
        )

    def create_edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._create_task(
            "/v1/videos/text2video",
            {
                "model_name": self.config.model_edit,
                "prompt": f"{payload['prompt']} Keep non-target regions unchanged.",
                "duration": str(self.config.duration),
                "mode": "std",
            },
            payload,
        )

    def _create_task(self, path: str, body: dict[str, Any], original_payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._jwt()}", "Content-Type": "application/json"}
        created = self.http.request_json("POST", f"{self.base_url}{path}", headers, body)
        data = created.get("data") or created
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise VideoApiError(f"Kling did not return task id: {created}")
        status = self._wait_task(path, task_id)
        result_data = status.get("data") or status
        task_result = result_data.get("task_result") or result_data.get("result") or {}
        videos = task_result.get("videos") or result_data.get("videos") or []
        video_url = videos[0].get("url") if videos and isinstance(videos[0], dict) else result_data.get("video_url")
        return self._uniform_response(
            original_payload,
            video_id=str(task_id),
            video_url=video_url,
            raw={"create": created, "status": status},
            metadata={"model": body.get("model_name")},
        )

    def _wait_task(self, path: str, task_id: str) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._jwt()}"}
        deadline = time.time() + self.config.timeout_seconds
        while time.time() < deadline:
            status = self.http.request_json("GET", f"{self.base_url}{path}/{task_id}", headers)
            data = status.get("data") or status
            state = data.get("task_status") or data.get("status")
            if state in {"succeed", "succeeded", "SUCCEEDED", "Success", "success"}:
                return status
            if state in {"failed", "FAILED", "Fail", "failure"}:
                raise VideoApiError(f"Kling task failed: {status}")
            time.sleep(self.config.poll_interval_seconds)
        raise VideoApiError(f"Kling task timed out: {task_id}")

    def _jwt(self) -> str:
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {"iss": self.access_key, "exp": now + 1800, "nbf": now - 5}
        signing_input = f"{self._b64(header)}.{self._b64(payload)}"
        signature = hmac.new(self.secret_key.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
        return f"{signing_input}.{self._b64_bytes(signature)}"

    @staticmethod
    def _b64(payload: dict[str, Any]) -> str:
        return KlingClient._b64_bytes(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    @staticmethod
    def _b64_bytes(payload: bytes) -> str:
        return base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")


def default_provider_config(provider: str, args: Any) -> ProviderConfig:
    defaults = {
        "hailuo": ("MiniMax-Hailuo-2.3", "MiniMax-Hailuo-2.3-Fast", "MiniMax-Hailuo-2.3", "768P"),
        "aliyun-wanx": ("wanx2.1-t2v-turbo", "wanx2.1-i2v-turbo", "wanx2.1-t2v-turbo", "720P"),
        "kling": ("kling-v2-1", "kling-v2-1", "kling-v2-1", "720P"),
    }
    t2v, i2v, edit, resolution = defaults.get(provider, defaults["hailuo"])
    return ProviderConfig(
        provider=provider,
        model_t2v=getattr(args, "t2v_model", None) or t2v,
        model_i2v=getattr(args, "i2v_model", None) or i2v,
        model_edit=getattr(args, "edit_model", None) or edit,
        duration=getattr(args, "duration", None) or 6,
        resolution=getattr(args, "resolution", None) or resolution,
        poll_interval_seconds=getattr(args, "poll_interval_seconds", None) or 5,
        timeout_seconds=getattr(args, "timeout_seconds", None) or 900,
        reference_image_url=getattr(args, "reference_image_url", None) or env("EVOVIDEO_REFERENCE_IMAGE_URL"),
        process_video=not getattr(args, "skip_video_processing", False),
        video_output_dir=getattr(args, "video_output_dir", None) or "outputs/provider_videos",
        sample_frames=getattr(args, "sample_frames", None) or 6,
    )


def provider_from_env(provider: str, args: Any) -> ProviderVideoClient:
    selected = select_auto_provider() if provider == "auto" else provider
    config = default_provider_config(selected, args)
    if selected == "hailuo":
        return HailuoClient(config)
    if selected == "aliyun-wanx":
        return AliyunWanxClient(config)
    if selected == "kling":
        return KlingClient(config)
    raise VideoApiError(f"unsupported provider: {provider}")


def select_auto_provider() -> str:
    if env("MINIMAX_API_KEY"):
        return "hailuo"
    if env("DASHSCOPE_API_KEY"):
        return "aliyun-wanx"
    if env("KLING_ACCESS_KEY") and env("KLING_SECRET_KEY"):
        return "kling"
    raise VideoApiError("No provider credentials found. Set MINIMAX_API_KEY, DASHSCOPE_API_KEY, or KLING_ACCESS_KEY/KLING_SECRET_KEY.")
