import re


ALLOWED_MESSAGE_RE = re.compile(r"[^a-zA-Z0-9 \u00C0-\u024F.,?!'\-]+")

NO_RESULTS_MESSAGE = (
    "We have some beautiful silver jewelry options that might interest you. "
    "I specialize in traditional silver jewelry like bangles, jhumkas, necklaces, and more. "
    "Would you like to explore our collection?"
)

OFF_TOPIC_MESSAGE = (
    "I can help you find silver jewelry from our collection. "
    "Try asking for items like jhumkas, bangles, necklaces, pendants, or sets."
)
