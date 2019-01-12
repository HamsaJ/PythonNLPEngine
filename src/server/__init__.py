from quart import Quart
from quart_cors import cors
from routes import routesBlueprint


server = Quart(__name__)
server = cors(server)
server.register_blueprint(routesBlueprint)
