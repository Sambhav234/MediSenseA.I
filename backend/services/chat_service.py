def detect_intent(msg: str) -> str:
    m = msg.lower().strip()

    if m in ["hi", "hello", "hey"]:
        return "greeting"

    if "help" in m:
        return "help"

    return "general"


def build_response(intent, msg):
    if intent == "greeting":
        return "Hello! 👋 I can help explain your symptoms or results."

    if intent == "help":
        return "Use Disease Predictor for diagnosis. I explain results."

    return "Tell me your prediction results or ask health questions."