import json
import typing
from dataclasses import dataclass


class JSONObject:
    def __init__(self, *args, **kwargs):
        # A hack to avoid stupid warnings about the init method of child classes not implemented.
        if args and args == kwargs:
            pass
        raise NotImplementedError("This is a 'abstract' Base Class. No direct instance can be created from it.")

    def __iter__(self):
        """Needed for dict(...) method call."""
        yield from vars(self).items()

    def to_dict(self):
        return json.loads(
            json.dumps(self, default=lambda o: getattr(o, '__dict__', str(o)))
        )

    # TODO: add optional support
    @classmethod
    def from_dict(cls, __dict: dict):
        if __dict.keys() == cls.__annotations__.keys():
            return cls(**__dict)

        for __type in cls.__annotations__.values():
            if generics := typing.get_args(__type):
                if len(generics) > 1:
                    raise TypeError(f"Type {__type} has more than one generic type. {generics}")
                __type = generics[0]
            if __dict.keys() == vars(__type).get("__dataclass_fields__", {}).keys():
                return __type.from_dict(__dict)
        raise ValueError(f"Could not creat Object from dict.")


@dataclass
class Data(JSONObject):
    text: str


@dataclass
class Dataset(JSONObject):
    version: str
    data: list[Data]


def main():
    #print(vars(Dataset)['__dataclass_fields__']['data'])
    s = """
    {
        "version": 1,
        "data": [
            {"text": "meep"},
            {"text": "yeet"}
        ]
    }
    """
    d: Dataset = json.loads(s, object_hook=Dataset.from_dict)
    print(d.to_dict())
    for data in d.data:
        data.text = data.text.upper()
        print(data.text)
    print(d.to_dict())



if __name__ == '__main__':
    main()
