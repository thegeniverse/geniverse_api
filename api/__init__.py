from ast import arguments
import base64
import requests
import uuid
import os
from io import BytesIO
from threading import Thread

from flask import Flask, request
from PIL import Image

from generation_utils import GenerationManager

from auth import create_api_key_guard, create_file_token_reader

generation_manager = GenerationManager()
app = Flask(__name__)

token_reader = create_file_token_reader()
api_key_required = create_api_key_guard(
    {
        "request": request,
        "token_reader": token_reader,
        "jwt_secret": os.environ["JWT_SECRET_KEY"],
    }
)


@app.route(
    "/generate",
    methods=["POST"],
)
@api_key_required
def generate(auth_data):
    print(f"auth_data = {auth_data}")

    try:
        prompt_list = request.form.get("text").split("-")

        auto = request.form.get("auto")
        if auto is None:
            auto = True
        else:
            auto = bool(int(auto))

        num_generations = request.form.get("numGenerations")
        if num_generations is None:
            num_generations = 1
        else:
            num_generations = int(num_generations)

        cond_img = request.form.get("condImg")
        if cond_img is not None:
            response = requests.get(cond_img)
            cond_img = Image.open(BytesIO(response.content)).convert("RGB")

        if auto:
            param_dict = None
        else:
            resolution = request.form.get("resolution").split(",")
            if resolution is None:
                resolution = (400, 400)
            else:
                resolution = [int(res) for res in resolution]

            strength = request.form.get("strength")
            if strength is None:
                strength = 0.3
            else:
                strength = float(strength)

            num_iterations = request.form.get("numIterations")
            if num_iterations is None:
                num_iterations = 30
            else:
                num_iterations = int(num_iterations)

            do_upscale = request.form.get("numIterations")
            if do_upscale is None:
                do_upscale = False
            else:
                do_upscale = bool(int(do_upscale))

            num_crops = request.form.get("realism")
            if num_crops is None:
                num_crops = 64
            else:
                num_crops = int(num_crops)

            param_dict = {
                "resolution": resolution,
                "lr": strength,
                "num_iterations": num_iterations,
                "do_upscale": do_upscale,
                "num_crops": num_crops,
            }

        print("param dict", param_dict)

        user_id = str(uuid.uuid4())
        _ = generation_manager.start_job(
            user_id=user_id,
            prompt_list=prompt_list,
            num_nfts=num_generations,
            cond_img=cond_img,
            auto=auto,
            param_dict=param_dict,
        )

        result_dict = {
            "success": True,
            "id": user_id,
        }

    except Exception as e:
        result_dict = {
            "success": False,
            "error": repr(e),
        }

    return result_dict


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


app.run(
    host="0.0.0.0",
    port=8100,
)
