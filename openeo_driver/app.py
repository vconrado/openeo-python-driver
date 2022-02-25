from openeo_driver.views import build_app

app = build_app()
app.run(port=9007)