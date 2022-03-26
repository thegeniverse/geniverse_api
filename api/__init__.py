import logging
import requests
import uuid
import os
from io import BytesIO

from flask import Flask, Response, make_response, request
from flask_cors import CORS
from PIL import Image

from generation_utils import GenerationManager

from auth import create_api_key_guard, create_file_token_reader

token_reader = create_file_token_reader()
api_key_required = create_api_key_guard(
    {
        "request": request,
        "token_reader": token_reader,
        "jwt_secret": os.environ.get("JWT_SECRET_KEY"),
    }
)


def create_app():
    generation_manager = GenerationManager()
    app = Flask(__name__)
    CORS(app)

    @app.route(
        "/generate",
        methods=["POST"],
    )
    @api_key_required
    def generate(auth_data):
        print(f"auth_data = {auth_data}")
        request_body = request.json

        assert request_body is not None, "where's your data bro?"
        assert "text" in request_body.keys(), "where's your god damn text?"

        # Generate new User ID per request
        user_id = str(uuid.uuid4())

        # Try downloading the client-provided conditioning image
        try:
            img_url = request_body.get("condImg", None)

            if img_url is None:
                return Response(response="condImg has to be defined", status=401)

            conditioning_img = Image.open(BytesIO(requests.get(img_url).content))
        except Exception as e:
            logging.error(e)
            return Response(
                response="Could not download conditioning image", status=400
            )

        # Configure job based on request form data
        job_configuration = {
            "user_id": user_id,
            "prompt_list": request_body.get("text").split("-"),
            "num_nfts": int(request_body.get("numGenerations", 1)),
            "auto": bool(int(request_body.get("auto", False))),
            "cond_img": conditioning_img,
            "param_dict": {
                "resolution": [
                    int(d) for d in request_body.get("resolution", "400,400").split(",")
                ],
                "lr": float(request_body.get("strength", 0.3)),
                "num_iterations": int(request_body.get("numIterations", 30)),
                "do_upscale": bool(int(request_body.get("do_upscale", False))),
                "num_crops": int(request_body.get("realism", 64)),
            },
        }

        # Try to start generation job
        logging.info(f"Starting new job:\n{job_configuration=}")
        try:
            generation_manager.start_job(**job_configuration)
        except Exception as e:
            make_response({"error": repr(e)}, 500)
        else:
            return {
                "success": True,
                "id": user_id,
            }

    @app.route(
        "/status",
        methods=["GET"],
    )
    def status():
        user_id = request.args.get("userId")

        status = generation_manager.get_user_status(user_id)

        results = None
        if status == "Done":
            results = generation_manager.get_user_results(user_id)

        return {
            "status": status,
            "results": results,
        }

    return app
