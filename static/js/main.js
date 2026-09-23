// Wizard & Form Interactivity for Wellbeing Check

let currentStep = 1;
const totalSteps = 3;

document.addEventListener("DOMContentLoaded", function () {
    var form = document.getElementById("assessment-form");
    if (!form) return;

    // Remove invalid style when user interacts with field
    var allInputs = form.querySelectorAll("input, select");
    allInputs.forEach(function (input) {
        input.addEventListener("input", function () {
            this.classList.remove("is-invalid");
        });
        input.addEventListener("change", function () {
            this.classList.remove("is-invalid");
        });
    });

    // Check if server validation returned errors on load
    var hasServerErrors = document.querySelector(".alert-danger");
    if (hasServerErrors) {
        var firstInvalid = form.querySelector(".is-invalid") || form.querySelector("input:invalid, select:invalid");
        if (firstInvalid) {
            var stepParent = firstInvalid.closest(".wizard-step");
            if (stepParent) {
                var stepId = parseInt(stepParent.id.replace("step-", ""));
                if (stepId) currentStep = stepId;
            }
        }
    }

    updateWizardState();
});

function updateWizardState() {
    for (let i = 1; i <= totalSteps; i++) {
        var stepElem = document.getElementById("step-" + i);
        var pillElem = document.getElementById("step-pill-" + i);
        if (stepElem) {
            if (i === currentStep) {
                stepElem.classList.add("active");
            } else {
                stepElem.classList.remove("active");
            }
        }
        if (pillElem) {
            if (i === currentStep) {
                pillElem.classList.add("active");
                pillElem.classList.remove("completed");
            } else if (i < currentStep) {
                pillElem.classList.add("completed");
                pillElem.classList.remove("active");
            } else {
                pillElem.classList.remove("active", "completed");
            }
        }
    }

    var progressBar = document.getElementById("wizard-progress-bar");
    if (progressBar) {
        var pct = (currentStep / totalSteps) * 100;
        progressBar.style.width = pct + "%";
    }

    var btnPrev = document.getElementById("btn-prev");
    var btnNext = document.getElementById("btn-next");
    var btnSubmit = document.getElementById("btn-submit");

    if (btnPrev) btnPrev.style.display = currentStep > 1 ? "inline-block" : "none";
    if (btnNext) btnNext.style.display = currentStep < totalSteps ? "inline-block" : "none";
    if (btnSubmit) btnSubmit.style.display = currentStep === totalSteps ? "inline-block" : "none";
}

function validateCurrentStep() {
    var stepElem = document.getElementById("step-" + currentStep);
    if (!stepElem) return true;

    var inputs = stepElem.querySelectorAll("input[required], select[required]");
    var firstInvalid = null;

    inputs.forEach(function (input) {
        if (!input.value || input.value.trim() === "") {
            input.classList.add("is-invalid");
            if (!firstInvalid) firstInvalid = input;
        } else {
            input.classList.remove("is-invalid");
        }
    });

    if (firstInvalid) {
        firstInvalid.scrollIntoView({ behavior: "smooth", block: "center" });
        firstInvalid.focus();
        return false;
    }
    return true;
}

function changeStep(delta) {
    if (delta > 0) {
        if (!validateCurrentStep()) return;
    }
    var nextStep = currentStep + delta;
    if (nextStep >= 1 && nextStep <= totalSteps) {
        currentStep = nextStep;
        updateWizardState();
        window.scrollTo({ top: 0, behavior: "smooth" });
    }
}

function jumpToStep(step) {
    if (step < currentStep) {
        currentStep = step;
        updateWizardState();
        window.scrollTo({ top: 0, behavior: "smooth" });
    } else if (step > currentStep) {
        if (validateCurrentStep()) {
            currentStep = step;
            updateWizardState();
            window.scrollTo({ top: 0, behavior: "smooth" });
        }
    }
}
