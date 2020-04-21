def normalize_timex(expression):
    u = expression.split()[1]
    v_input = float(expression.split()[0])

    if u in ["instantaneous", "forever"]:
        return u, str(1)

    convert_map = {
        "seconds": 1.0,
        "minutes": 60.0,
        "hours": 60.0 * 60.0,
        "days": 24.0 * 60.0 * 60.0,
        "weeks": 7.0 * 24.0 * 60.0 * 60.0,
        "months": 30.0 * 24.0 * 60.0 * 60.0,
        "years": 365.0 * 24.0 * 60.0 * 60.0,
        "decades": 10.0 * 365.0 * 24.0 * 60.0 * 60.0,
        "centuries": 100.0 * 365.0 * 24.0 * 60.0 * 60.0,
    }
    seconds = convert_map[u] * float(v_input)
    prev_unit = "seconds"
    for i, v in enumerate(convert_map):
        if seconds / convert_map[v] < 0.1:
            break
        prev_unit = v
    if prev_unit == "seconds" and seconds > 60.0:
        prev_unit = "centuries"

    return prev_unit


print(normalize_timex("40 years"))
print(normalize_timex("55 years"))
print(normalize_timex("100 years"))
print(normalize_timex("200 years"))

