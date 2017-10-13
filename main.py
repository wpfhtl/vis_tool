from flask import Flask ,session,url_for
from flask import app as bpp
from app_page import app
from feature_page import feature_page
from saliency_page import saliency_page
import os
main = Flask(__name__)

if main.debug:
    main.secret_key = '...' ##secret_key
else:
    main.secret_key = os.urandom(24)
"""
register your own blueprint here
"""
main.register_blueprint(app)
main.register_blueprint(feature_page,url_prefix='/feature_page') 
main.register_blueprint(saliency_page,url_prefix='/saliency_page')
if __name__ == "__main__":
    main.run(host='0.0.0.0', debug=True, threaded=True)

