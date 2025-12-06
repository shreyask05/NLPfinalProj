#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import requests
import re
from collections import Counter


INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data

API_KEY = "cse476"
API_BASE = "http://10.4.58.53:41701/v1"
MODEL = "bens_model"
def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

#function to classify the kind of question that is being asked (domain in the dev dataset)
########################## CONSTANTS OUTSIDE THE MAIN CODE BLOCKS ###################################
router_labels = {
    "math",
    "coding",
    "future_prediction",
    "planning",
    "common_sense"
}
#the prompt for the LLM to figure out what kind of question is being asked
router_layer_prompt = '''
You are a question classifier.

You will need to do the following:
- Read the input question
- Figure out which one of the following domains best describes the question
- Only reply with the domain name, nothing else

This is the list of valid domains, and a short description of the characteristics of the domain:
- math: questions that require any mathematical calculation, equations, inequalities, or any numerical reasoning.
- coding: question that ask about programs, code, algorithms, or debugging code.
- future_prediction: questions that ask about what will happen in the future, creating forecasts, or hypothetical events.
- planning: questions about making plans, schedules, or step-by-step strategies.
- common_sense: everyday reasoning, intuitive judgements, or logic questions that do not require math or coding to solve.

Always respond with only the name of the domain, which is one of:
math, coding, future_prediction, planning, common_sense
'''.strip()

########################## START OF THE METHODS USED BY THE AGENT ###################################

def build_routing_question(question):
    prompt = f"""
    Question:
    {question}

    Classify this question into one of the following domains:
    math, coding, future_prediction, planning, common_sense

    Reply with only the domain name.
    """.strip()
    return prompt


#building the full prompt
def classify_question(question):
    system_prompt = router_layer_prompt
    question_prompt = build_routing_question(question)

    response = call_model_chat_completions(
        prompt=question_prompt,
        system=system_prompt,
        temperature=0.0,
        timeout=5
    )
    domain = (response.get("text") or "").lower().strip()
    if domain in router_labels:
        return domain

    for label in router_labels:
        if label in domain:
            return label

    #worst case repsonse if nothing matches
    return "common_sense"


'''         These are the domain specific prompts for each of the router labels that were defined above         '''
def build_domain_system_prompt(domain):
    """
    math prompt idea:
    - use CoT for better thinking process
    - use self-consistency to ensure that the majority of models converge on a single answer
    - then format final answer after the majority ans is decided
    """
    sys_prompt = (
        "You are a helpful assistant.\n"
        "Follow the user's question prompt exactly.\n"
        "Take time to reason in a step-by-step manner internally.\n"
        "The output should be returned only in the format requested, with no thought process or explanation.\n"
    )
    if domain == "math":
        sys_prompt = (
            "You are the best mathematical assistant.\n"
            "For each math question, do the following:\n"
            "- Think about the math problem step-by-step.\n"
            "- Carefully check the algebra, arithmetic, and logic done during each step\n"
            "- Make sure the the solution meets the criteria listed in the question.\n"
            "- If the answer is a number, output just that number."
            "If it is an expression, output just that expression.\n"
            "Do NOT include phrases like 'The answer is' or any reasoning."
        )
    elif domain == "coding":
        sys_prompt = (
            "You are an expert software engineer.\n"
            "You MUST follow these rules for every coding problem:\n"
            "1. Read the problem carefully and understand the required behavior.\n"
            "2. Plan the solution internally (in your head) before you write code.\n"
            "3. Output ONLY the final code solution, with no explanation or comments.\n"
            "4. Do NOT wrap the code in markdown fences (no ```), just plain code.\n"
            "5. Match the language and any format specified in the problem statement "
            "(for example, if it says 'in Python', write Python code).\n"
            "6. If a specific function signature or class name is given, use it exactly.\n"
        )
    elif domain in {"future_prediction", "planning", "common_sense"}:
        sys_prompt = (
            "You are a precise reasoning assistant.\n"
            "You will think through the question carefully and internally, "
            "and then output ONLY the final answer in the requested format "
            "(usually a single word, short phrase, or number). "
            "Do NOT show your reasoning."
        )
        if domain == "future_prediction":
            sys_prompt = (
                "You are a precise commonsense reasoner.\n"
                "For each question, decide whether the correct answer is yes or no.\n"
                "Internally, think it through carefully.\n"
                "Then output ONLY:\n"
                "  true   if the correct answer is yes\n"
                "  false  if the correct answer is no\n"
                "Use lowercase, no punctuation, no extra words."
                "Only answer true or false.\n"
            )
        elif domain == "planning":
            sys_prompt += "\nFocus on producing a clear, actionable plan or decision."

        elif domain == "common_sense":
            sys_prompt += "\nFocus on everyday commonsense and intuitive reasoning."

    return sys_prompt

########################## MATH DOMAIN SPECIFIC CoT + Self Consistency ###################################

def call_math_once(question, temp):
    system_prompt  = build_domain_system_prompt("math")
    response = call_model_chat_completions(
        prompt=question,
        system = system_prompt,
        temperature = temp
    )
    return (response.get("text") or "").strip()

def math_formatting(answer):
    answer = answer.strip().lower()
    answer = re.sub(r"[^0-9\-\/\.]", "", answer)
    return answer

def math_self_consistency(question, polls, temp):
    answers = []
    for _ in range(polls):
        curr_answer = call_math_once(question, temp)
        answers.append(curr_answer)
    return answers

def math_majority_vote(answers):
    if not answers:
        return ""
    formatted = [math_formatting(ans) for ans in answers if ans.strip()]
    if not formatted:
        return answers[0]

    votes = Counter(formatted)
    best_match,_ = votes.most_common(1)[0]

    for ans in answers:
        if math_formatting(ans) == best_match:
            return ans

    return answers[0]


def extract_final_answer_math(text: str) -> str:
    s = text.strip()

    m = re.findall(r"####\s*([^\n]+)", s)
    if m:
        return m[-1].strip()

    frac_matches = re.findall(r"\\frac\{([^}]+)\}\{([^}]+)\}", s)
    if frac_matches:
        num, den = frac_matches[-1]
        return f"\\frac{{{num}}}{{{den}}}"

    num_matches = re.findall(r"-?\d+(?:\.\d+)?", s)
    if num_matches:
        return num_matches[-1]

    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return s

def math_finalize(answer):
    prompt = (
        f"Current answer: {answer}\n"
        "Is this answer in its simplest form. If it is a fraction, try to reduce it to an integer if possible"
    )
    system_prompt = "Your job is to reduce and simplify math answers if possible. Only simplify answers if possible"
    response = call_model_chat_completions(
        prompt=prompt,
        system = system_prompt,
        temperature = 0
    )
    final_answer = (response.get("text") or "").strip()
    return final_answer

def math_cot(question, curr_ans):
    system_prompt = (
        "You are a mathematical validator.\n"
        "Your job is to verify a proposed answer to a math question.\n"
        "You need to think step-by-step to decide whether the answer is valid for the question.\n"
        "If the proposed answer is correct, then only output that answer with no change.\n"
        "If it is incorrect, then recompute the correct answer, and only output the correct answer.\n"
        "In any case, only output the single final answer, expression, or format requested\n"
        "No explanations, no sentences, no thought process, no testing, only the final answer."
    )
    prompt = (
        f"Problem: \n {question}\n"
        f"Proposed answer: {curr_ans}\n"
        "Carefully verify whether the answer is valid or not.\n"
        "If needed, recompute and output the correct answer.\n"
    )

    response = call_model_chat_completions(
        prompt=prompt,
        system = system_prompt,
        temperature = 0
    )
    cot_checked = (response.get("text") or "").strip()
    return cot_checked


def math_main(question):
    answers = math_self_consistency(question, polls=3, temp=0.3)
    majority_vote = math_majority_vote(answers)
    simplified = math_finalize(majority_vote)
    CoT_check = math_cot(question, simplified)
    return extract_final_answer_math(CoT_check)



########################## CODING DOMAIN SPECIFIC SYSTEM PROMPTS ###################################

def call_coding_once(question, temp):
    system_prompt  = build_domain_system_prompt("coding")
    response = call_model_chat_completions(
        prompt=question,
        system = system_prompt,
        temperature = temp
    )
    return (response.get("text") or "").strip()

def coding_main(question):
    answer = call_coding_once(question, temp=0)
    return answer


########################## REASONING DOMAIN SPECIFIC (self-consistency used here for common sense questions) ###################################

'''----------------------- Common sense methods ------------------------'''
def common_sense_booleans(answer):
    answer = answer.strip().lower()
    if "true" in answer or "yes" in answer:
        return "true"
    elif "false" in answer or "no" in answer:
        return "false"
    else:
        return ""

def common_sense_main(question, polls=3):
    system_prompt = build_domain_system_prompt("common_sense")
    answers = []

    for _ in range(polls):
        response = call_model_chat_completions(
            prompt=question,
            system = system_prompt,
            temperature = 0.3
        )
        answer = (response.get("text") or "").strip()
        answers.append(answer)

    votes = Counter(common_sense_booleans(answer) for answer in answers)
    votes.pop("", None)

    if not votes:
        return answers[0]

    best_match,_ = votes.most_common(1)[0]
    for answer in answers:
        if common_sense_booleans(answer) == best_match:
            return answer

    return answers[0]

'''---------------------- Simple llm call for planning and prediction domains ----------------------'''
def planning_predictions(domain, question):
    system_prompt = build_domain_system_prompt(domain)
    response = call_model_chat_completions(
        prompt=question,
        system = system_prompt,
        temperature = 0
    )
    answer = (response.get("text") or "").strip()
    return answer

'''---------------------- Main method for al three reasoning domains ----------------------'''
def reasoning_main(domain, question):
    if domain == "common_sense":
        return common_sense_main(question=question)
    elif domain in {"future_prediction", "planning"}:
        return planning_predictions(domain=domain, question=question)
    else:
        return planning_predictions(domain="common_sense", question=question)



########################## METHOD TO ROUTE QUESTIONS TO RESPECTIVE MAINS ###################################

'''---------------------- routing the question to specific process after the classification is done ----------------------'''
def route(domain: str, question: str) -> str:
    if domain == "math":
        return math_main(question)
    if domain == "coding":
        return coding_main(question)
    if domain in {"common_sense", "planning", "future_prediction"}:
        return reasoning_main(domain, question)
    return reasoning_main("common_sense", question)

def agent_loop(question):
    domain = classify_question(question)
    answer = route(domain, question)
    if answer=="yes" or answer=="Yes":
        answer = "true"
    if answer=="no" or answer=="No":
        answer = "false"
    return answer


def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers = []
    count = 0
    for idx, question in enumerate(questions, start=1):
        print(f"### working on question {count} ###")
        count += 1
        # Example: assume you have an agent loop that produces an answer string.
        real_answer = agent_loop(question["input"])
        answers.append({"output": real_answer})
        #placeholder_answer = f"Placeholder answer for question {idx}"
        #answers.append({"output": placeholder_answer})
    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions)

    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()

