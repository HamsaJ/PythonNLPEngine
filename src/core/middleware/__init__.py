class CorsMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, scope):
        for header, value in scope["headers"]:
            if header == "X-Secret" and value == "very-secret":
                return self.app(scope)
        return self.error_response

    def error_response(self, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": ["content-length", "0"],
            }
        )
        await send(
            {"type": "http.response.body", "body": b"", "more_body": False,}
        )
