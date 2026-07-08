import os
from flask import Flask
from dotenv import load_dotenv
from flask_cors import CORS

# 🔥 Load environment variables
load_dotenv()

# Import services & routes
from backend.services.prediction_service import PredictionService
from backend.services.skin_service import SkinService
from backend.services.xray_service import XRayService
from backend.routes.predict import predict_bp
from backend.routes.secondary import secondary_bp


def create_app():
    app = Flask(__name__)

    # 🔥 Enable CORS (VERY IMPORTANT)
    CORS(app)

    # ML model path
    ml_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "ml",
        "models"
    )

    # Load prediction service
    pred_service = PredictionService(ml_dir)
    xray_service = XRayService()
    skin_service = SkinService()

    # Store in app config
    app.config["PRED_SVC"] = pred_service
    app.config["XRAY_SVC"] = xray_service
    app.config["SKIN_SVC"] = skin_service

    # Register routes
    app.register_blueprint(secondary_bp)
    app.register_blueprint(predict_bp)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
