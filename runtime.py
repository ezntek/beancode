def bean_input():
    def is_integer(val):
        if len(val) == 0:
            return False
        for ch in val:
            if not ch.isdigit():
                return False
        return True
    def is_real(val):
        if len(val) == 0:
            return False
        if is_integer(val):
            return False
        found_decimal = False
        for ch in val:
            if ch == ".":
                if found_decimal:
                    return False
                found_decimal = True
        return found_decimal
    inp = input() 
    if is_real(inp):
        return float(inp)
    elif is_integer(inp):
        return int(inp)
    elif (t := inp.strip().lower()) in {"true", "false", "no", "yes"}:
        return True if t in {"true", "yes"} else False
    return inp

