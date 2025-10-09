# app/api/middleware/monitor_service.py
import csv, time
from datetime import datetime, timezone
from uuid import uuid4
from typing import Awaitable, Callable
from fastapi import Request, Response

csv_header = [
    "request_id",
    "datetime",
    "endpoint_triggered",
    "client_ip_address",
    "response_time_ms",
    "status_code",
    "is_successful",
]

async def monitor_service(req: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    request_id = uuid4().hex   
    request_datetime = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    response: Response = await call_next(req)
    response_time = round(time.perf_counter() - start_time, 4)
    response.headers["X-Response-Time"] = str(response_time)
    response.headers["X-API-Request-ID"] = request_id
    with open("usage_monitoring.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(csv_header)
        writer.writerow(
            [
                request_id,
                request_datetime,
                str(req.url),
                req.client.host if req.client else "unknown",
                response_time,
                response.status_code,
                response.status_code < 400,
            ]
        )
    return response
