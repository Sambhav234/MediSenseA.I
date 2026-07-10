import os
from flask import Flask
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Import services & routes
from backend.services.prediction_service import PredictionService
from backend.services.skin_service import SkinService
from backend.services.xray_service import XRayService
from backend.routes.predict import predict_bp
from backend.routes.secondary import secondary_bp
from backend.config import Config


def create_app():

    app = Flask(__name__)

    # Load production configuration
    app.config.from_object(Config)

    # Enable CORS
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": "*"
            }
        }
    )

    # ML model directory
    ml_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "ml",
        "models"
    )

    # Initialize services
    app.config["PRED_SVC"] = PredictionService(ml_dir)
    app.config["XRAY_SVC"] = XRayService()
    app.config["SKIN_SVC"] = SkinService()

    # Register routes
    app.register_blueprint(predict_bp)
    app.register_blueprint(secondary_bp)

    # Health endpoint
    @app.get("/health")
    def health():

        return {
            "backend": "running",
            "prediction_model": app.config["PRED_SVC"].ready,
            "xray_model": app.config["XRAY_SVC"].ready,
            "skin_model": app.config["SKIN_SVC"].ready,
        }, 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )