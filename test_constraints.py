from services import extract_constraints

print('=== FINAL TEST RESULTS ===')
print()
tests = [
    ('Budget 5000', '', 5000),
    ('under 500', '', 500),
    ('my budget is 2000', '', 2000),
    ('5000', 'What is your budget?', 5000),
    ('budget around 1500', '', 1500),
    ('rs 800', '', 800),
]

for msg, context, expected in tests:
    result = extract_constraints(msg, context)
    actual = result.get('max_price', 'NONE')
    status = 'PASS' if actual == expected else 'FAIL'
    print(f'{status}: "{msg}" -> max_price: {actual} (expected: {expected})')
