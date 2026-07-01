"""
evaluate_traces.py — Recall@10 Evaluation against all 10 conversation traces (C1–C10)

For each trace:
  1. Replay the FULL conversation (all user turns) as a single API call.
  2. Check how many of the expected final-turn URLs appear in the agent's recommendations.
  3. Compute Recall@10 = hits / total_expected for each trace.
  4. Report mean Recall@10 across all traces.

Run with:
    python evaluate_traces.py
"""

import sys
import time
import json
from fastapi.testclient import TestClient
from main import app

# ============================================================================
# Ground-truth data extracted from C1–C10 final recommendation turns
# ============================================================================

TRACES = [
    {
        "id": "C1",
        "description": "Senior leadership / CXO selection with OPQ",
        "messages": [
            {"role": "user", "content": "We need a solution for senior leadership."},
            {"role": "assistant", "content": "Happy to help narrow that down. Who is this meant for?"},
            {"role": "user", "content": "The pool consists of CXOs, director-level positions; people with more than 15 years of experience."},
            {"role": "assistant", "content": "For such roles, the OPQ32r is the right instrument. Is this for a new position or developmental feedback?"},
            {"role": "user", "content": "Selection — comparing candidates against a leadership benchmark."},
        ],
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
            "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
            "https://www.shl.com/products/product-catalog/view/opq-leadership-report/",
        ],
    },
    {
        "id": "C2",
        "description": "Senior Rust engineer — catalog gap + cognitive test",
        "messages": [
            {"role": "user", "content": "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?"},
            {"role": "assistant", "content": "SHL's catalog doesn't include a Rust-specific knowledge test. The closest fits are Smart Interview Live Coding, Linux Programming, and Networking and Implementation. Want me to build a shortlist?"},
            {"role": "user", "content": "Yes, go ahead. Should I also add a cognitive test for this level?"},
        ],
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
            "https://www.shl.com/products/product-catalog/view/linux-programming-general/",
            "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/",
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    {
        "id": "C3",
        "description": "Entry-level contact centre agents, English US",
        "messages": [
            {"role": "user", "content": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"},
            {"role": "assistant", "content": "Before I shape the stack — what language are the calls in?"},
            {"role": "user", "content": "English."},
            {"role": "assistant", "content": "SVAR has four English variants: US, UK, Australian, and Indian. Which fits your operation?"},
            {"role": "user", "content": "US. New simulation for volume, old solution for finalists. Confirmed."},
        ],
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
            "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
            "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
            "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
        ],
    },
    {
        "id": "C4",
        "description": "Graduate financial analysts — numerical + situational judgement",
        "messages": [
            {"role": "user", "content": "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test."},
            {"role": "assistant", "content": "For graduate-level financial analysts, I recommend numerical reasoning, financial accounting, statistics, and OPQ32r."},
            {"role": "user", "content": "Good. Can you also add a situational judgement element — work-context decision making for graduates?"},
            {"role": "assistant", "content": "Added Graduate Scenarios."},
            {"role": "user", "content": "That covers it. Numerical + Graduate Scenarios as first filter, domain tests for shortlisted candidates."},
        ],
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
            "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
            "https://www.shl.com/products/product-catalog/view/basic-statistics-new/",
            "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    {
        "id": "C5",
        "description": "Sales organization re-skilling audit",
        "messages": [
            {"role": "user", "content": "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?"},
            {"role": "assistant", "content": "For a compact audit-and-development stack: GSA, OPQ32r, OPQ MQ Sales Report, and Sales Transformation."},
            {"role": "user", "content": "What's the difference between OPQ and OPQ MQ Sales Report?"},
            {"role": "assistant", "content": "OPQ32r is the underlying personality questionnaire. The OPQ MQ Sales Report summarizes OPQ results in a sales-specific way."},
            {"role": "user", "content": "Clear. We'll use OPQ for everyone and add MQ only where we want motivators in the Sales Report; keeping the five solutions as our audit stack."},
        ],
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/global-skills-assessment/",
            "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
            "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/",
            "https://www.shl.com/products/product-catalog/view/salestransformationreport2-0-individualcontributor/",
        ],
    },
    {
        "id": "C6",
        "description": "Chemical plant operators — safety-critical, industrial context",
        "messages": [
            {"role": "user", "content": "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?"},
            {"role": "assistant", "content": "For a safety-critical role, I recommend DSI, Safety and Dependability 8.0, and Workplace Health and Safety."},
            {"role": "user", "content": "What's the difference between the DSI and the Safety & Dependability 8.0?"},
            {"role": "assistant", "content": "DSI is standalone; the 8.0 has industrial norms. If your facility is industrial-classified, the 8.0 is better."},
            {"role": "user", "content": "We're industrial. The 8.0 bundle is the right fit. Confirmed."},
        ],
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/safety-and-dependability-focus-8-0/",
            "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/",
        ],
    },
    {
        "id": "C7",
        "description": "Bilingual healthcare admin — hybrid English/Spanish battery",
        "messages": [
            {"role": "user", "content": "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?"},
            {"role": "assistant", "content": "There is a catalog constraint: role-specific knowledge tests are English-only but personality measures support Spanish. Hybrid or personality-only?"},
            {"role": "user", "content": "They're functionally bilingual — English fluent for written work. Go with the hybrid."},
            {"role": "assistant", "content": "Full hybrid battery: HIPAA, Medical Terminology, Microsoft Word, DSI, OPQ32r."},
            {"role": "user", "content": "Are we legally required under HIPAA to test all staff who touch patient records?"},
            {"role": "assistant", "content": "Those are legal questions outside what I can advise on. Your legal team is the right resource."},
            {"role": "user", "content": "Understood. Keep the shortlist as-is."},
        ],
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/hipaa-security/",
            "https://www.shl.com/products/product-catalog/view/medical-terminology-new/",
            "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/",
            "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    {
        "id": "C8",
        "description": "Admin assistants — Excel and Word with simulations",
        "messages": [
            {"role": "user", "content": "I need to quickly screen admin assistants for Excel and Word daily."},
            {"role": "assistant", "content": "For a quick knowledge check I recommend MS Excel (New), MS Word (New), and OPQ32r. Say the word if you'd prefer to skip personality."},
            {"role": "user", "content": "In that case, I am OK with adding a simulation - we want to capture the capabilities."},
            {"role": "assistant", "content": "Updated with simulations: Microsoft Excel 365 (New) and Microsoft Word 365 (New)."},
            {"role": "user", "content": "That's good."},
        ],
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
            "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/",
            "https://www.shl.com/products/product-catalog/view/ms-excel-new/",
            "https://www.shl.com/products/product-catalog/view/ms-word-new/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    {
        "id": "C9",
        "description": "Senior full-stack engineer (backend-leaning, 7 turns)",
        "messages": [
            {"role": "user", "content": "Senior Full-Stack Engineer JD: 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker. Can you recommend an assessment battery?"},
            {"role": "assistant", "content": "Is this a backend-leaning role or frontend-heavy?"},
            {"role": "user", "content": "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional."},
            {"role": "assistant", "content": "Is the seniority closer to a senior IC or a tech lead?"},
            {"role": "user", "content": "Senior IC. They lead design on their own services but don't manage other engineers directly."},
            {"role": "assistant", "content": "Shortlist: Core Java Advanced, Spring, RESTful Web Services, SQL, Verify G+, OPQ32r."},
            {"role": "user", "content": "Add AWS and Docker. Drop REST — the API design signal will come through in Spring and the live interview."},
            {"role": "assistant", "content": "Updated: REST out, AWS and Docker in."},
            {"role": "user", "content": "Keep Verify G+. Locking it in."},
        ],
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
            "https://www.shl.com/products/product-catalog/view/spring-new/",
            "https://www.shl.com/products/product-catalog/view/sql-new/",
            "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/",
            "https://www.shl.com/products/product-catalog/view/docker-new/",
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        ],
    },
    {
        "id": "C10",
        "description": "Graduate management trainees — OPQ dropped by user",
        "messages": [
            {"role": "user", "content": "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates."},
            {"role": "assistant", "content": "For a graduate management trainee battery: Verify G+, OPQ32r, Graduate Scenarios."},
            {"role": "user", "content": "Can you remove the OPQ32r and replace it with something shorter? Candidates complain it takes too long."},
            {"role": "assistant", "content": "OPQ32r is the most relevant solution. There is no shorter alternative as a replacement."},
            {"role": "user", "content": "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios."},
        ],
        "expected_urls": [
            "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
            "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
        ],
    },
]


def recall_at_k(recommended_urls: list, expected_urls: list, k: int = 10) -> float:
    """Compute Recall@K: fraction of expected URLs found in top-K recommendations."""
    if not expected_urls:
        return 0.0
    top_k = set(url.rstrip("/") for url in recommended_urls[:k])
    expected = set(url.rstrip("/") for url in expected_urls)
    hits = len(top_k & expected)
    return hits / len(expected)


def run_evaluation():
    print("=" * 65)
    print("  SHL Assessment Recommender — Recall@10 Evaluation")
    print("=" * 65)
    print(f"  Traces: {len(TRACES)} | Top-K: 10")
    print("=" * 65)

    results = []

    with TestClient(app) as client:
        for trace in TRACES:
            trace_id = trace["id"]
            description = trace["description"]
            messages = trace["messages"]
            expected_urls = trace["expected_urls"]

            print(f"\n[{trace_id}] {description}")
            print(f"  Turns: {len(messages)} | Expected: {len(expected_urls)} assessments")

            try:
                # Send the FULL conversation history in one stateless call
                payload = {"messages": messages}
                response = client.post("/chat", json=payload, timeout=120)

                if response.status_code != 200:
                    print(f"  ERROR: HTTP {response.status_code} — {response.text[:200]}")
                    results.append({"id": trace_id, "recall": 0.0, "hits": 0, "expected": len(expected_urls), "error": True})
                    time.sleep(5)
                    continue

                data = response.json()
                recommended_urls = [rec["url"] for rec in data.get("recommendations", [])]
                recall = recall_at_k(recommended_urls, expected_urls, k=10)

                # Count hits
                top_k = set(url.rstrip("/") for url in recommended_urls[:10])
                expected_set = set(url.rstrip("/") for url in expected_urls)
                hits = len(top_k & expected_set)

                results.append({
                    "id": trace_id,
                    "recall": recall,
                    "hits": hits,
                    "expected": len(expected_urls),
                    "recommended": len(recommended_urls),
                    "error": False,
                })

                status = "PASS" if recall >= 0.5 else "FAIL"
                print(f"  [{status}] Recall@10: {recall:.2f} ({hits}/{len(expected_urls)} hits)")

                # Show which URLs were found / missed
                for url in expected_urls:
                    found = url.rstrip("/") in top_k
                    mark = "HIT " if found else "MISS"
                    name = url.split("/view/")[1].rstrip("/") if "/view/" in url else url
                    print(f"    [{mark}] {name}")

                if recommended_urls:
                    print(f"  Agent returned {len(recommended_urls)} recommendations")
                else:
                    print(f"  Agent returned: {data.get('reply', '')[:100]}...")

                # Rate limit buffer between calls
                time.sleep(3)

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"id": trace_id, "recall": 0.0, "hits": 0, "expected": len(expected_urls), "error": True})
                time.sleep(10)

    # ---- Summary ----
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Trace':<6} {'Recall@10':<12} {'Hits':<10} {'Status'}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*6}")

    valid_results = [r for r in results if not r.get("error")]
    for r in results:
        status = "ERROR" if r.get("error") else ("PASS" if r["recall"] >= 0.5 else "FAIL")
        hits_str = f"{r['hits']}/{r['expected']}" if not r.get("error") else "N/A"
        recall_str = f"{r['recall']:.2f}" if not r.get("error") else "N/A"
        print(f"  {r['id']:<6} {recall_str:<12} {hits_str:<10} {status}")

    if valid_results:
        mean_recall = sum(r["recall"] for r in valid_results) / len(valid_results)
        total_hits = sum(r["hits"] for r in valid_results)
        total_expected = sum(r["expected"] for r in valid_results)
        print(f"\n  Mean Recall@10 : {mean_recall:.4f} ({mean_recall*100:.1f}%)")
        print(f"  Total Hits     : {total_hits}/{total_expected}")
        print(f"  Traces Passing : {sum(1 for r in valid_results if r['recall'] >= 0.5)}/{len(valid_results)}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_evaluation()
