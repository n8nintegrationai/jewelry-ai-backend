from services import extract_constraints, get_previous_context

print('=== FINAL TEST RESULTS ===')
print()
tests = [
    ('Budget 5000', '', {'max_price': 5000, 'is_price_filter': True}),
    ('under 500', '', {'max_price': 500, 'is_price_filter': True}),
    ('my budget is 2000', '', {'max_price': 2000, 'is_price_filter': True}),
    ('5000', 'What is your budget?', {'max_price': 5000, 'is_price_filter': True}),
    ('budget around 1500', '', {'max_price': 1500, 'is_price_filter': True}),
    ('rs 800', '', {'max_price': 800, 'is_price_filter': True}),
    ('ruby earrings under 1000', '', {'max_price': 1000, 'is_price_filter': False}),
]

for msg, context, expected in tests:
    result = extract_constraints(msg, context)
    actual = {
        'max_price': result.get('max_price', 'NONE'),
        'is_price_filter': result.get('is_price_filter', False),
    }
    status = 'PASS' if actual == expected else 'FAIL'
    print(f'{status}: "{msg}" -> {actual} (expected: {expected})')

print()
history = [
    {'role': 'user', 'content': 'ruby earrings'},
    {'role': 'assistant', 'content': 'Here are some ruby earrings (499)'},
    {'role': 'user', 'content': 'under 1000'},
]
context = get_previous_context(history)
expected_context = {
    'last_category': 'ruby earrings',
    'last_query': 'ruby earrings',
    'last_max_price': 1000,
}
actual_context = {
    'last_category': context.get('last_category'),
    'last_query': context.get('last_query'),
    'last_max_price': context.get('last_max_price'),
}
status = 'PASS' if actual_context == expected_context else 'FAIL'
print(f"{status}: history context -> {actual_context} (expected: {expected_context})")
