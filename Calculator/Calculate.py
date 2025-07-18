import os
from .Build import build
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.numbers import Float, ComplexInfinity

def calculate(input_filedir='Calculator/images/canvas_image.png',model_path='Calculator/model25.pth'):
    for i in os.listdir('Calculator/output_fig'):
        os.remove(os.path.join('Calculator/output_fig', i))
    string1=build(input_filedir,model_path)
    string1 = string1.replace("times", "*")
    string1 = string1.replace("div", "/")
    converted = string1
    if len(converted) == 0:
        return ""
    if converted[-1] == "=":
        converted = converted[0:len(converted)-1]
    converted = converted.replace("=", "==")
    converted = converted.replace(")(", ")*(")
    converted = converted.replace("^", "**")
    try:
        parsed=parse_expr(converted)
        evaluated = parsed.evalf() if not isinstance(parsed, bool) else parsed
        if type(evaluated) == Float:
            result = "%g"%(evaluated)
        elif type(evaluated) == ComplexInfinity:
            result = "div/0!"
        else:
            result = evaluated
        return string1, result
    except SyntaxError as e:
        return string1, "?"
    except TypeError as e:
        return string1, "?"
    except Exception as e:
        return string1, "?"