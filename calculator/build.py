"""Inline models.json and predict.js into the page template -> index.html"""
tpl = open("calculator.template.html", encoding="utf-8").read()
tpl = tpl.replace("/*MODELS_JSON*/", open("models.json", encoding="utf-8").read().strip())
tpl = tpl.replace("/*PREDICT_JS*/", open("predict.js", encoding="utf-8").read().replace("if (typeof module", "if (false && typeof module"))
open("index.html", "w", encoding="utf-8").write(tpl)
print("index.html", round(len(tpl) / 1e6, 2), "MB")
