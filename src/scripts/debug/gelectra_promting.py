from src.qa.gelectra import Gelectra
from src.qa.quad import QUAD

dataset = QUAD.GermanQUAD.TEST
datapoint = dataset.data[0][0]
context = datapoint.context
question = datapoint.qas[0].question
answers = [answer.text for answer in datapoint.qas[0].answers]

print(f"{context=}\n{question=}\n{answers}")

GelectraGermanQuad = Gelectra.GermanQuad
output = GelectraGermanQuad.prompt(question, context)
print(f"text = '{output['text']}' \n Span = '{output['span']}")