from threading import Timer
import re


# ==================================================================#
# Decorator to prevent a function's actions from being run until
# at least x seconds have passed without the function being called
# ==================================================================#
def debounce(wait):
    def decorator(fun):
        def debounced(*args, **kwargs):
            def call_it():
                fun(*args, **kwargs)

            try:
                debounced.t.cancel()
            except AttributeError:
                pass

            debounced.t = Timer(wait, call_it)
            debounced.t.start()

        return debounced

    return decorator


# ==================================================================#
# Replace fancy quotes and apostrope's with standard ones
# ==================================================================#
def fixquotes(txt):
    txt = txt.replace("“", '"')
    txt = txt.replace("”", '"')
    txt = txt.replace("’", "'")
    txt = txt.replace("`", "'")
    return txt


# ==================================================================#
# 
# ==================================================================#
def trimincompletesentence(txt):
    # Cache length of text
    ln = len(txt)
    # Find last instance of punctuation (Borrowed from Clover-Edition by cloveranon)
    lastpunc = max(txt.rfind("."), txt.rfind("!"), txt.rfind("?"))
    # Is this the end of a quote?
    if lastpunc < ln - 1:
        if txt[lastpunc + 1] == '"':
            lastpunc = lastpunc + 1
    if lastpunc >= 0:
        txt = txt[:lastpunc + 1]
    return txt


# ==================================================================#
# 
# ==================================================================#
def replaceblanklines(txt):
    txt = txt.replace("\n\n", "\n")
    return txt


# ==================================================================#
# 
# ==================================================================#
def removespecialchars(txt, variables=None):
    if variables is None or variables.actionmode == 0:
        txt = re.sub(r"[#/@%<>{}+=~|\^]", "", txt)
    else:
        txt = re.sub(r"[#/@%{}+=~|\^]", "", txt)
    return txt


# ==================================================================#
# If the next action follows a sentence closure, add a space
# ==================================================================#
def addsentencespacing(txt, variables):
    txt = singlelineprocessing(txt, variables, True)
    return txt


def singlelineprocessing(txt, variables, spacing=False):
    txt = variables.regex_sl.sub('', txt)
    if len(variables.actions) > 0:
        if len(variables.actions[variables.actions.get_last_key()]) > 0:
            action = variables.actions[variables.actions.get_last_key()]
            lastchar = action[-1] if len(action) else ""
        else:
            # Last action is blank, this should never happen, but
            # since it did let's bail out.
            return txt
    else:
        action = variables.prompt
        lastchar = action[-1] if len(action) else ""
    if lastchar != "\n":
        txt = txt + "\n"
    if (
            lastchar == "." or lastchar == "!" or lastchar == "?" or lastchar == "," or lastchar == ";" or lastchar == ":") and spacing:
        txt = " " + txt
    return txt


# ==================================================================#
#  Cleans string for use in file name
# ==================================================================#
def cleanfilename(filename):
    filteredcharacters = ('/', '\\')
    filename = "".join(c for c in filename if c not in filteredcharacters).rstrip()
    return filename


# ==================================================================#
#  Newline substitution for fairseq models
# ==================================================================#
def encodenewlines(txt, newlinemode):
    if newlinemode == "s":
        return txt.replace('\n', "</s>")
    return txt


def decodenewlines(txt, newlinemode):
    if newlinemode == "s":
        return txt.replace("</s>", '\n')
    return txt
