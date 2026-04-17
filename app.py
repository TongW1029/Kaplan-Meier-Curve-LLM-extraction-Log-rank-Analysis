import base64
import json
import math
import os
import urllib.error
import urllib.request
import warnings
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
import cgi


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
HOST = os.environ.get("HOST", "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))


SAMPLE_EXTRACTION = {
    "study_title": "Demo Kaplan-Meier Extraction",
    "note": "This is a demo dataset for classroom presentation.",
    "groups": [
        {
            "name": "Treatment A",
            "observations": [
                {"time": 2, "event": 1},
                {"time": 3, "event": 1},
                {"time": 4, "event": 0},
                {"time": 5, "event": 1},
                {"time": 6, "event": 1},
                {"time": 7, "event": 0},
                {"time": 8, "event": 1},
                {"time": 9, "event": 1},
            ],
        },
        {
            "name": "Treatment B",
            "observations": [
                {"time": 2, "event": 1},
                {"time": 3, "event": 0},
                {"time": 4, "event": 1},
                {"time": 5, "event": 0},
                {"time": 6, "event": 1},
                {"time": 7, "event": 1},
                {"time": 8, "event": 1},
                {"time": 9, "event": 1},
            ],
        },
    ],
}


def json_response(handler, payload, status=HTTPStatus.OK):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def text_response(handler, text, status=HTTPStatus.OK, content_type="text/plain; charset=utf-8"):
    data = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_json_body(handler):
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(content_length) if content_length > 0 else b"{}"
    return json.loads(raw.decode("utf-8"))


def validate_extraction(payload):
    if not isinstance(payload, dict):
        raise ValueError("The extracted result must be a JSON object.")

    groups = payload.get("groups")
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError("At least two groups are required.")

    cleaned_groups = []
    for group in groups[:2]:
        if not isinstance(group, dict):
            raise ValueError("Each group must be a JSON object.")
        name = str(group.get("name", "Unknown Group")).strip() or "Unknown Group"
        observations = group.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"Group '{name}' has no observations.")

        cleaned_observations = []
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError(f"Group '{name}' contains an invalid observation.")
            time_value = float(observation.get("time"))
            if time_value < 0:
                raise ValueError("Time cannot be negative.")
            event_value = int(observation.get("event"))
            if event_value not in (0, 1):
                raise ValueError("Event must be 0 or 1.")
            cleaned_observations.append({"time": time_value, "event": event_value})

        cleaned_groups.append({"name": name, "observations": cleaned_observations})

    return {
        "study_title": str(payload.get("study_title", "Untitled Study")),
        "note": str(payload.get("note", "")),
        "groups": cleaned_groups,
    }


def group_summary(group):
    observations = group["observations"]
    total = len(observations)
    events = sum(item["event"] for item in observations)
    censored = total - events
    max_time = max(item["time"] for item in observations) if observations else 0.0
    return {
        "name": group["name"],
        "total": total,
        "events": events,
        "censored": censored,
        "max_time": max_time,
    }


def compute_km_step_points(observations):
    ordered = sorted(observations, key=lambda item: (item["time"], -item["event"]))
    event_times = sorted({item["time"] for item in ordered if item["event"] == 1})
    points = [{"time": 0.0, "survival": 1.0}]
    survival = 1.0

    for event_time in event_times:
        at_risk = sum(1 for item in ordered if item["time"] >= event_time)
        events = sum(1 for item in ordered if item["time"] == event_time and item["event"] == 1)
        if at_risk == 0:
            continue
        points.append({"time": event_time, "survival": survival})
        survival *= (1.0 - events / at_risk)
        points.append({"time": event_time, "survival": survival})

    return points


def compute_log_rank(group_a, group_b):
    obs_a = group_a["observations"]
    obs_b = group_b["observations"]
    event_times = sorted({item["time"] for item in obs_a + obs_b if item["event"] == 1})

    observed_a = 0.0
    expected_a = 0.0
    variance = 0.0
    rows = []

    for event_time in event_times:
        n_a = sum(1 for item in obs_a if item["time"] >= event_time)
        n_b = sum(1 for item in obs_b if item["time"] >= event_time)
        d_a = sum(1 for item in obs_a if item["time"] == event_time and item["event"] == 1)
        d_b = sum(1 for item in obs_b if item["time"] == event_time and item["event"] == 1)
        n_total = n_a + n_b
        d_total = d_a + d_b

        if n_total == 0 or d_total == 0:
            continue

        expected = d_total * (n_a / n_total)
        observed_a += d_a
        expected_a += expected

        if n_total > 1:
            variance_piece = (
                n_a
                * n_b
                * d_total
                * (n_total - d_total)
                / ((n_total ** 2) * (n_total - 1))
            )
        else:
            variance_piece = 0.0
        variance += variance_piece

        rows.append(
            {
                "time": event_time,
                "risk_a": n_a,
                "risk_b": n_b,
                "events_a": d_a,
                "events_b": d_b,
                "expected_a": round(expected, 4),
                "variance_piece": round(variance_piece, 6),
            }
        )

    if variance <= 0:
        z_score = 0.0
        chi_square = 0.0
        p_value = 1.0
    else:
        z_score = (observed_a - expected_a) / math.sqrt(variance)
        chi_square = z_score ** 2
        p_value = math.erfc(math.sqrt(chi_square / 2.0))

    return {
        "observed_a": round(observed_a, 4),
        "expected_a": round(expected_a, 4),
        "variance": round(variance, 6),
        "z_score": round(z_score, 4),
        "chi_square": round(chi_square, 4),
        "p_value": round(p_value, 6),
        "decision": "statistically significant" if p_value < 0.05 else "not statistically significant",
        "table": rows,
    }


def bucher_indirect_from_ci(hr_ab, low_ab, high_ab, hr_bc, low_bc, high_bc):
    if min(hr_ab, low_ab, high_ab, hr_bc, low_bc, high_bc) <= 0:
        raise ValueError("HR and confidence interval values must all be positive.")

    log_hr_ab = math.log(hr_ab)
    log_hr_bc = math.log(hr_bc)
    se_ab = (math.log(high_ab) - math.log(low_ab)) / (2 * 1.96)
    se_bc = (math.log(high_bc) - math.log(low_bc)) / (2 * 1.96)

    log_hr_ac = log_hr_ab + log_hr_bc
    se_ac = math.sqrt(se_ab ** 2 + se_bc ** 2)
    z_score = log_hr_ac / se_ac if se_ac > 0 else 0.0
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))

    return {
        "hr_ac": round(math.exp(log_hr_ac), 4),
        "low_ac": round(math.exp(log_hr_ac - 1.96 * se_ac), 4),
        "high_ac": round(math.exp(log_hr_ac + 1.96 * se_ac), 4),
        "z_score": round(z_score, 4),
        "p_value": round(p_value, 6),
    }


def extract_output_text(response_json):
    if isinstance(response_json.get("output_text"), str) and response_json["output_text"].strip():
        return response_json["output_text"]

    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]

    raise ValueError("The API response did not contain model text output.")


def build_extraction_prompt(group_a_name, group_b_name):
    return (
        "You are helping with a biostatistics class project. "
        "Visually inspect the Kaplan-Meier survival curve image and reconstruct an approximate "
        "patient-level dataset for exactly two groups, but stay as faithful to the figure as possible. "
        f"If the legend is hard to read, use the fallback group names '{group_a_name}' and '{group_b_name}'. "
        "If the legend is readable, prefer the real group names shown in the figure. "
        "Return only structured data. "
        "For each group, create a list of observations, where each observation has: "
        "'time' (numeric follow-up time) and 'event' (1 if event happened, 0 if censored). "
        "Important rules: "
        "1) Use the x-axis unit and range from the figure, and keep all times within that range. "
        "2) Use visible step drops as event times and visible tick marks on the curve as censored observations. "
        "3) If a number-at-risk table is shown, use it to keep the sample size and censoring pattern roughly consistent. "
        "4) Do not invent arbitrary fine-grained times unless the figure strongly supports them. "
        "5) The total number of observations in each group should be close to the initial number at risk if that value is visible. "
        "6) Prefer approximate curve-faithful reconstruction over merely plausible synthetic data. "
        "Also provide a short note describing uncertainties in the reconstruction."
    )


def call_llm_extract(image_bytes, filename, api_key, base_url, model, group_a_name, group_b_name):
    mime_type = "image/png"
    lowered = filename.lower()
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        mime_type = "image/jpeg"
    elif lowered.endswith(".webp"):
        mime_type = "image/webp"
    elif lowered.endswith(".gif"):
        mime_type = "image/gif"

    image_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    schema = {
        "type": "object",
        "properties": {
            "study_title": {"type": "string"},
            "note": {"type": "string"},
            "groups": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "observations": {
                            "type": "array",
                            "minItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "time": {"type": "number"},
                                    "event": {"type": "integer", "enum": [0, 1]},
                                },
                                "required": ["time", "event"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["name", "observations"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["study_title", "note", "groups"],
        "additionalProperties": False,
    }

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_extraction_prompt(group_a_name, group_b_name)},
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "km_reconstruction",
                "strict": True,
                "schema": schema,
            }
        },
    }

    endpoint = base_url.rstrip("/") + "/responses"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM API request failed: {exc.code} {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach the LLM API: {exc.reason}") from exc

    response_text = extract_output_text(response_json)
    parsed = json.loads(response_text)
    return validate_extraction(parsed)


class KMHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            index_file = WEB_DIR / "index.html"
            text_response(self, index_file.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")
            return

        if self.path == "/api/sample":
            json_response(self, SAMPLE_EXTRACTION)
            return

        text_response(self, "Not Found", status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path == "/api/analyze":
            try:
                payload = validate_extraction(read_json_body(self))
                group_a, group_b = payload["groups"][0], payload["groups"][1]
                result = {
                    "study_title": payload["study_title"],
                    "note": payload["note"],
                    "summaries": [group_summary(group_a), group_summary(group_b)],
                    "km_curves": [
                        {"name": group_a["name"], "points": compute_km_step_points(group_a["observations"])},
                        {"name": group_b["name"], "points": compute_km_step_points(group_b["observations"])},
                    ],
                    "log_rank": compute_log_rank(group_a, group_b),
                }
                json_response(self, result)
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/indirect":
            try:
                payload = read_json_body(self)
                result = bucher_indirect_from_ci(
                    float(payload["hr_ab"]),
                    float(payload["low_ab"]),
                    float(payload["high_ab"]),
                    float(payload["hr_bc"]),
                    float(payload["low_bc"]),
                    float(payload["high_bc"]),
                )
                json_response(self, result)
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path == "/api/extract":
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    },
                )

                image_item = form["image"] if "image" in form else None
                if image_item is None or getattr(image_item, "file", None) is None:
                    raise ValueError("Please upload an image first.")

                image_bytes = image_item.file.read()
                filename = image_item.filename or "km_curve.png"
                if not image_bytes:
                    raise ValueError("Uploaded image is empty.")

                api_key = form.getfirst("api_key", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
                if not api_key:
                    raise ValueError("OPENAI_API_KEY is missing. Paste it into the form or set it in your environment.")

                base_url = form.getfirst("base_url", "").strip() or "https://api.openai.com/v1"
                model = form.getfirst("model", "").strip() or "gpt-4.1-mini"
                group_a_name = form.getfirst("group_a_name", "").strip() or "Treatment A"
                group_b_name = form.getfirst("group_b_name", "").strip() or "Treatment B"

                extraction = call_llm_extract(
                    image_bytes=image_bytes,
                    filename=filename,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    group_a_name=group_a_name,
                    group_b_name=group_b_name,
                )
                json_response(self, extraction)
            except Exception as exc:
                json_response(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        text_response(self, "Not Found", status=HTTPStatus.NOT_FOUND)

    def log_message(self, format_text, *args):
        return


def main():
    WEB_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), KMHandler)
    print(f"Server running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
