from flask import Flask, render_template, request, redirect, url_for, session

import config
from database import init_db, save_assessment, get_history, clear_history
from predict import make_prediction
from utils import validate_form_data

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# Create database table on startup
init_db()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/assess", methods=["GET", "POST"])
def assess():
    if request.method == "GET":
        return render_template("assess.html")

    # Validate form input
    cleaned, errors = validate_form_data(request.form)
    if errors:
        return render_template("assess.html", errors=errors, form_data=request.form)

    # Run prediction
    result = make_prediction(cleaned)

    # Save to database
    save_assessment(
        result["score"],
        result["risk_level"]["label"],
        result["dimensions"],
    )

    # Store result in session for the result page
    session["last_result"] = result

    return redirect(url_for("result"))


@app.route("/result")
def result():
    result = session.get("last_result")
    if not result:
        return redirect(url_for("assess"))
    return render_template("result.html", result=result)


@app.route("/history")
def history():
    assessments = get_history()
    return render_template("history.html", assessments=assessments)


@app.route("/clear-history", methods=["POST"])
def handle_clear_history():
    clear_history()
    return redirect(url_for("history"))


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=config.DEBUG, port=5000)
