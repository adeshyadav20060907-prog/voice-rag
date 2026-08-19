import requests
import statistics
import time

URL = "http://127.0.0.1:8000/ask"

QUESTIONS = [
    "What is the capital of India?",
    "What is Moore's Law?",
    "What is Eroom's Law?",
    "What is a transistor?",
    "Explain artificial intelligence.",
    "भारत की राजधानी क्या है?",
    "मूर का नियम क्या है?",
    "ट्रांजिस्टर क्या है?",
    "भारताची राजधानी काय आहे?",
    "गुजरातची राजधानी कोणती आहे?"
]


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


def run_test(question):
    start = time.perf_counter()

    try:
        response = requests.post(
            URL,
            json={
                "question": question,
                "language": "English"
            },
            timeout=60
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        if response.status_code != 200:
            return {
                "question": question,
                "latency": elapsed,
                "success": False,
                "answer": ""
            }

        data = response.json()

        return {
            "question": question,
            "latency": elapsed,
            "success": True,
            "answer": data.get(
                "answer",
                ""
            )
        }

    except Exception as e:

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        print(
            "ERROR:",
            repr(e)
        )

        return {
            "question": question,
            "latency": elapsed,
            "success": False,
            "answer": ""
        }


def main():

    print()
    print("======================================")
    print("VOICE RAG EVALUATION")
    print("======================================")
    print()

    results = []

    for index, question in enumerate(
        QUESTIONS,
        start=1
    ):

        print(
            f"[{index}/{len(QUESTIONS)}] "
            f"{question}"
        )

        result = run_test(
            question
        )

        results.append(result)

        print(
            "Latency:",
            round(
                result["latency"],
                2
            ),
            "ms"
        )

        print(
            "Success:",
            result["success"]
        )

        print()

    successful = [
        result["latency"]
        for result in results
        if result["success"]
    ]

    if not successful:

        print(
            "No successful requests."
        )

        return

    p50 = percentile(
        successful,
        0.50
    )

    p70 = percentile(
        successful,
        0.70
    )

    p100 = max(
        successful
    )

    average = statistics.mean(
        successful
    )

    success_rate = (
        len(successful)
        / len(results)
    ) * 100

    print()
    print("======================================")
    print("LATENCY RESULTS")
    print("======================================")

    print(
        "Successful:",
        len(successful),
        "/",
        len(results)
    )

    print(
        "Success rate:",
        round(
            success_rate,
            2
        ),
        "%"
    )

    print(
        "Average:",
        round(
            average,
            2
        ),
        "ms"
    )

    print(
        "P50:",
        round(
            p50,
            2
        ),
        "ms"
    )

    print(
        "P70:",
        round(
            p70,
            2
        ),
        "ms"
    )

    print(
        "P100:",
        round(
            p100,
            2
        ),
        "ms"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()