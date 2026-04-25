# This file acts as an alias in case the deployment start command is set to "gunicorn cortexa:app"
from app import app

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
