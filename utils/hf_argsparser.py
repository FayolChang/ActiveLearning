import json
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import NewType, Any, Union, Iterator

import dataclasses

DataClass = NewType('DataDlass', Any)
DataClassType = NewType('DataClassType', Any)


class HfArgumentParser(ArgumentParser):

    dataclass_types: Iterator[DataClassType]

    def __init__(self, dataclass_types: Union[DataClassType, Iterator[DataClassType]], **kwargs):
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = dataclass_types
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for field in dataclasses.fields(dtype):
            field_name = f'--{field.name}'
            kwargs = field.metadata.copy()

            typestring = str(field.type)
            for x in (int, float, str):
                if typestring == f'typing.Union[{x.__name__}, NoneType]':
                    field.type = x

            if isinstance(field.type,type) and issubclass(field.type, Enum):
                kwargs['choices'] = list(field.type)
                kwargs['type'] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs['default'] = field.default
            elif field.type is bool:
                kwargs['action'] = 'store_false' if field.default is True else 'store_true'
                if field.default is True:
                    field_name = f'--no_{field.name}'
                    kwargs['dest'] = field.name
            else:
                kwargs['type'] = field.type
                if field.default is not dataclasses.MISSING:
                    kwargs['default'] = field.default
                else:
                    kwargs['required'] = True

            self.add_argument(field_name, kwargs)

    def parse_args_into_dataclasses(self, args=None, return_remaining_strings=False):
        if len(sys.argv):
            args_file = Path(sys.argv[0]).with_suffix('.args')
            print(args_file)
            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args else fargs + sys.argv[1:]
        namespace, remaining_args = self.parse_known_args(args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)

        if len(namespace.__dict__) > 0:
            outputs.append(namespace)

        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            return (*outputs, )

    def parse_json_file(self, json_file):
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype)}
            inputs = {k: v for k, v in data.items() if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        return (*outputs, )









