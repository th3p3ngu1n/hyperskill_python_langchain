# Write your solution below
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_groq import ChatGroq
import dotenv, os

dotenv.load_dotenv()

def get_planet_examples() -> list[dict[str, str]]:
    planet_examples = []
    planets_dir = "planets"
    for entry in os.listdir(planets_dir):
        with open(os.path.join(planets_dir, entry), "r", encoding="utf-8") as file:
            input_name = file.readline().strip()
            planet_examples.append({"input": input_name, "output": file.read().replace("\n", " ").strip()})
    return planet_examples

def stage_2():
    planet_examples = get_planet_examples()

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_retries=2,)
    planet = input()
    example_template = PromptTemplate.from_template("Q: {input}\nA: {output}")

    few_shot_prompt = FewShotPromptTemplate(
        examples=planet_examples,
        example_prompt=example_template,
        suffix="Q: {planet}\nA:",
        input_variables=["planet"],
    )

    final_prompt = few_shot_prompt.format(planet=planet)
    response = llm.invoke(final_prompt)
    print(response.content)

def stage_1():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, max_retries=2,)
    question = input()
    template = PromptTemplate.from_template("You are a helpful assistant who answers questions user may have. You are asked: {question}.")

    prompt = template.invoke({"question": question})
    response = llm.invoke(prompt)
    print(response.content)

if __name__ == "__main__":
    stage_2()
