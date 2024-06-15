import json
import pandas as pd
from openai import OpenAI


def retrieve_QA_from_context(context):
    prompt = get_context() + context
    client = OpenAI()
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an IB-math teacher, skilled in generating IB syllabus-aligned SL math question."},
                {"role": "user","content": prompt}]
        )
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
    JSON string. The format of the returned string including "Question returned" and "Answer returned" should be as follows,
    {
        "Question": "Question returned",
        "Answer": "Answer returned"
    }
    . The context is as follows: 
    '''

if __name__ == '__main__':
    context = '''
        numbers and algebra
        '''
    df = retrieve_QA_from_context(context)

    print(df.to_string())

