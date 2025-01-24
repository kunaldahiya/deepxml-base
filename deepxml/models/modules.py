import re
import json
import torch.nn as nn
from ._modules import ELEMENTS


def parse_json(file, ARGS):
    with open(file, encoding='utf-8') as f:
        file = ''.join(f.readlines())
        schema = resolve_schema_args(file, ARGS)
    # The documentation says that it preserves order (3.6+)
    return json.loads(schema)


def _construct_module(key, args):
    # TODO: Check for extendibility
    try:
        return ELEMENTS[key](**args)
    except KeyError:
        raise NotImplementedError("Unknown module!!")


def construct_module(config):
    if len(config) == 0:
        return _construct_module('identity', {})
    elif len(config) == 1:
        k, v = next(iter(config.items()))
        return _construct_module(k, v)
    else:
        modules = [_construct_module(k, v) for k, v in config.items()] 
        return nn.Sequential(*modules)


def resolve_schema_args(jfile, ARGS):
    arguments = re.findall(r"#ARGS\.(.+?);", jfile)
    for arg in arguments:
        replace = '#ARGS.%s;' % (arg)
        to = str(ARGS.__dict__[arg])
        # Python True and False to json true & false
        if to == 'True' or to == 'False':
            to = to.lower()
        if jfile.find('\"#ARGS.%s;\"' % (arg)) != -1:
            replace = '\"#ARGS.%s;\"' % (arg)
            if isinstance(ARGS.__dict__[arg], str):
                to = str("\""+ARGS.__dict__[arg]+"\"")
        jfile = jfile.replace(replace, to)
    return jfile

 