from src.qa.qamodel import QAModel

m: QAModel = QAModel.EnglishQA
r = m.prompt("What do we call this?", "This is what we call a test.")
print(r.text)