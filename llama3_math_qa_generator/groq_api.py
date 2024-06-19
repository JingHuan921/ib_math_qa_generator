import os
from groq import Groq
from typing import Dict, List
import json
import pandas as pd
from store_api_key import api_key

def system(content: str):
    return { "role": "system", "content": content }

def user(content: str):
    return { "role": "user", "content": content }

def chat_completion(
    messages: List[Dict],
    model = "llama3-70b-8192",
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:

    os.environ["GROQ_API_KEY"] = api_key
    client = Groq()
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    return response
def retrieve_QA_from_context(context):
    prompt = get_context() + context

    response = chat_completion(messages=[
        system("You are an IB-math teacher, skilled in generating IB syllabus-aligned SL math question."),
        user(prompt)
    ])
    content = response.choices[0].message.content


    try:
        data = json.loads(content)
        return pd.DataFrame(data.items(), columns=["IB-Math", "Generated"])

    except (json.JSONDecodeError, IndexError):
        pass

    return pd.DataFrame({
        "IB-Math": ["Question", "Workings + Answer"],
        "Generated": ["", ""]
    })


def get_context():
    return '''
    Generate an IB syllabus-aligned SL math question, with difficulty level as medium, 
    and type as non-calculator on the topic context I append below, and its corresponding answer workings. If you can't find a suitable response, 
    please return '' and don't make things up. Always return your response as a valid 
    JSON string only, no preface before that. The format of the returned string should be as follows, replace the value of "Question" and "Answer" with generated result:
    {
        "Question": " The equation x^2 + px + q = 0 has roots α and β. If α + β = 5 and αβ = 2, find the values of p and q.",
        "Answer": "Since α + β = 5, we know that the sum of the roots is 5. Since αβ = 2, we know that the product of the roots is 2. Recall that the sum of the roots of a quadratic equation x^2 + px + q = 0 is -p, and the product of the roots is q. Therefore, -p = 5 and q = 2. Hence, p = -5 and q = 2."
    }
    . The context is as follows: 
    '''

if __name__ == '__main__':
    context = '''
        How do I find the probability of combined events?
The probability of A or B (or both) occurring can be found using the formula
straight P(A U B)= P(A) + P(B) - P(A ∩ B). You subtract the probability of A and B both
occurring because it has been included twice (once in P(A) and once in P(B) ).
The probability of A and B occurring can be found using the formula.
P(A ∩ B) = P(A) P(B|A)
        '''
    df = retrieve_QA_from_context(context)

    print(df.to_string())

