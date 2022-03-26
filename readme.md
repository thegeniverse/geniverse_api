# Geniverse API

A simple Python process that exposes a REST API that allows anyone
to create applications on top of the Geniverse core library. It
uses PyTorch and the Geniverse core libraries to expose generative
models such as VQGAN+CLIP.

## Usage

Running the flask server can be done like this:

```sh
export JWT_SECRET_KEY="JWT_SECRET"
export FLASK_APP="api"
export PYTHONPATH="api"

flask run
```

## Development

### Styling

The project uses the Python Black formatter and Flake8 as the
linter.
