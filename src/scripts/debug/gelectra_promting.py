import torch

from src.qa.gelectra import Gelectra
from src.qa.quad import QUAD

dataset = QUAD.GermanQUAD.TEST
datapoint = dataset.data[0][0]
context = datapoint.context
question = datapoint.qas[0].question
answers = [answer.text for answer in datapoint.qas[0].answers]

short_input = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur vel maximus mauris. Suspendisse mi velit, euismod eget dolor quis, lacinia mollis dui. Mauris convallis semper vestibulum. Suspendisse id magna quis risus mattis convallis ac eu neque. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam malesuada vehicula tellus eu efficitur. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos.
"""

long_input = """
Duis efficitur pharetra sapien, cursus fringilla lectus ultricies quis. Praesent nisi lectus, congue ac mauris et, tempor elementum velit. Donec pretium maximus consequat. Sed non libero at lorem ullamcorper tincidunt ut id diam. Nullam egestas enim ac erat tristique luctus. Nam eget condimentum nulla. Curabitur blandit, orci viverra mollis dictum, turpis augue tincidunt elit, quis sodales massa velit vel dolor. Nulla eleifend, nisl a egestas scelerisque, ipsum ligula sagittis magna, non consectetur tortor felis sit amet felis. Integer risus enim, molestie in ultricies vitae, cursus ut sem. Phasellus iaculis enim vel lobortis convallis.
Sed molestie laoreet arcu laoreet finibus. Curabitur maximus odio nec imperdiet elementum. Phasellus non lacinia tortor. Ut eget sem ut neque fermentum sagittis eu et augue. Nunc quis tellus at nisi pellentesque tempor ullamcorper at est. Mauris cursus lacinia mauris a ullamcorper. Sed id erat in tortor pretium cursus. Donec eget consequat leo, ut venenatis nulla. Aliquam erat volutpat. Praesent viverra ornare enim sed ullamcorper. Donec nisl dolor, mattis tincidunt tortor sed, dapibus tincidunt ipsum. Phasellus feugiat hendrerit diam, at placerat libero bibendum nec. Donec imperdiet enim eu felis hendrerit porta. Vivamus id massa quis turpis commodo dapibus. Sed sit amet nisi sit amet nibh ullamcorper tristique.
Nunc id porttitor massa. Phasellus dictum dui vel lectus suscipit vestibulum. Suspendisse faucibus auctor neque vitae ultricies. Vestibulum at condimentum lorem. Pellentesque tortor augue, porta et leo tristique, dictum auctor lorem. Suspendisse quis volutpat lacus. Phasellus id dapibus nulla. Cras bibendum libero dolor, ac eleifend risus tempor et. Phasellus lobortis molestie risus, bibendum hendrerit erat cursus nec. Nullam eget commodo velit, vel cursus ipsum. Phasellus vehicula mi vitae bibendum tempus. Vestibulum lobortis ultrices consequat. In ut ligula vulputate eros elementum accumsan ac et dolor. Suspendisse facilisis venenatis sem id semper. 
"""

print(f"{context=}\n{question=}\n{answers}")

model = Gelectra.GermanQuad

model_input_s = model.tokenizer.encode_qa(short_input, short_input)
with torch.no_grad():
    os = model.model(**Gelectra.filter_dict_for_model_input(model_input_s))

model_input_l = model.tokenizer.encode_qa(short_input, long_input)
with torch.no_grad():
    ol = model.model(**Gelectra.filter_dict_for_model_input(model_input_l))

on = model.prompt(question, context)
