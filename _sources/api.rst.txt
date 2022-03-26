*************
Geniverse API
*************

Welcome to the Geniverse API. You can use the Geniverse API to access state-of-the-art
generative artificial intelligence models.

Endpoints
#########

For now, the API only supports a single endpoint.

Generations
===========

``POST /generate``

This resource lets you schedule a generation job:

.. code-block:: sh

    curl -X POST localhost:8000/generation