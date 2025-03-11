from langsmith import wrappers, Client
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from typing import Dict, List
import re

# Load environment variables
load_dotenv()

# Initialize LangSmith client
client = Client()

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7
)


def target(inputs: dict) -> dict:
    response = llm.invoke([
        {"role": "system", "content": "Answer the following question accurately"},
        {"role": "user", "content": inputs["question"]}
    ])
    return {"response": response.content}
    
# Create teacher response examples
examples = [
    (
        "Could you tell me about a challenging classroom management situation you've faced?",
        """Last week, I had a particularly difficult situation with two students constantly disrupting the class. They would talk during lessons, distract others, and sometimes even argue with each other. I tried several approaches before finding what worked.

First, I separated them physically in the classroom. Then, I gave them specific responsibilities - one became responsible for collecting homework, the other for distributing materials. This gave them positive ways to channel their energy.

The most effective strategy was having a private conversation with each student separately. I learned that one was struggling with the material but was too embarrassed to ask for help, while the other was dealing with family issues at home.

After understanding their individual situations, I adjusted my approach. I provided extra support with the coursework and made sure to check in with them regularly. Within two weeks, their behavior improved significantly."""
    ),
    (
        "How do you handle students who struggle with motivation?",
        """In my experience, motivation issues often stem from deeper causes. I recently worked with a student who showed no interest in mathematics. Instead of immediately pushing more math problems, I took time to understand their perspective.

I discovered they felt mathematics was irrelevant to their interests in music. So, I started incorporating music-related examples - we discussed frequency, rhythm patterns, and sound waves. This connection helped them see mathematics in a new light.

I also implemented a reward system where students could earn points for improvement, not just correct answers. This shifted focus from pure achievement to personal growth.

The key was finding what personally resonated with each student and building bridges between their interests and the subject matter."""
    ),
    (
        "What's your approach to inclusive education in your classroom?",
        """Inclusive education is central to my teaching philosophy. In my current class, I have students with varying abilities, including two with learning disabilities and one gifted student.

I implement differentiated instruction by providing multiple ways to engage with the material. For example, during our literature unit, I offer audio books alongside traditional texts, use visual aids, and allow students to demonstrate understanding through various mediums - written, oral, or creative projects.

I've also established a buddy system where students support each other. This not only helps academically but also builds empathy and understanding among students.

Regular communication with parents and support staff ensures we're all aligned in supporting each student's unique needs."""
    ),
    (
        "How do you incorporate technology in your teaching?",
        """Technology integration has been transformative in my classroom. During the pandemic, I developed a hybrid approach that I continue to use. I utilize interactive online platforms for homework assignments and create video explanations for complex topics that students can review at their own pace.

However, I'm mindful of the digital divide. Not all students have equal access to technology at home, so I ensure essential learning can be completed with basic resources. I maintain a balance between digital and traditional teaching methods.

I've found tools like Kahoot for review sessions and Google Classroom for assignment management particularly effective. These platforms help track student progress and provide immediate feedback."""
    ),
    (
        "What's your strategy for maintaining classroom discipline?",
        """My approach to classroom discipline is proactive rather than reactive. At the beginning of each term, students collaboratively create our classroom rules. This ownership makes them more likely to follow and enforce these rules themselves.

When issues arise, I focus on understanding the root cause rather than just addressing the behavior. For instance, when a student was consistently late to class, instead of immediately punishing them, I learned they were helping their younger sibling get to school first.

I use a three-step approach: first a private conversation, then a documented warning, and finally parental involvement if needed. However, I find that building strong relationships and setting clear expectations prevents most disciplinary issues."""
    )
]

inputs = [{"question": input_prompt} for input_prompt, _ in examples]
outputs = [{"answer": output_answer} for _, output_answer in examples]

# Create dataset in LangSmith
dataset = client.create_dataset(
    dataset_name="Teacher Interview Responses",
    description="A collection of detailed teacher responses to common interview questions about classroom management and teaching methods."
)

# Add examples to the dataset
client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

# Define evaluator classes
class TeacherResponseEvaluators:
    @staticmethod
    def practical_examples(outputs: dict) -> Dict:
        """Evaluates the use of specific, practical examples"""
        instructions = """Rate the response on:
        1. Specific situation description (0-0.3)
        2. Clear action steps taken (0-0.4)
        3. Outcome description (0-0.3)
        
        Return score and identify key examples used."""
        
        response = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Response to evaluate: {outputs['response']}"}
        ])
        
        return {
            "score": float(re.search(r"(\d+\.?\d*)", response.content).group(1)),
            "examples": response.content
        }
    
    @staticmethod
    def reflection_depth(outputs: dict) -> Dict:
        """Evaluates the depth of reflection and self-awareness"""
        instructions = """Analyze the response for:
        1. Self-awareness (0-0.3)
        2. Critical thinking (0-0.3)
        3. Learning from experience (0-0.2)
        4. Professional growth (0-0.2)"""
        
        response = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Response to evaluate: {outputs['response']}"}
        ])
        
        return {
            "score": float(re.search(r"(\d+\.?\d*)", response.content).group(1)),
            "analysis": response.content
        }
    
    @staticmethod
    def student_centricity(outputs: dict) -> Dict:
        """Evaluates focus on student needs and outcomes"""
        instructions = """Rate how well the response demonstrates:
        1. Understanding of student needs (0-0.3)
        2. Adaptability to student differences (0-0.3)
        3. Focus on student outcomes (0-0.2)
        4. Evidence of student engagement (0-0.2)"""
        
        response = llm.invoke([
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Response to evaluate: {outputs['response']}"}
        ])
        
        return {
            "score": float(re.search(r"(\d+\.?\d*)", response.content).group(1)),
            "feedback": response.content
        }

# Run evaluation
experiment_results = client.evaluate(
    target,
    data="Teacher Interview Responses",
    evaluators=[
        TeacherResponseEvaluators.practical_examples,
        TeacherResponseEvaluators.reflection_depth,
        TeacherResponseEvaluators.student_centricity
    ],
    experiment_prefix="teacher-responses-eval",
    max_concurrency=2,
)

if __name__ == "__main__":
    print("Teacher response dataset created and evaluation started!")
    print(f"Dataset ID: {dataset.id}")
    print(f"Number of examples: {len(examples)}")
    print("\nEvaluations being run:")
    print("1. Practical Examples (with specific situations)")
    print("2. Reflection Depth (self-awareness and critical thinking)")
    print("3. Student Centricity (focus on student needs)")
    print("\nResults will be available in LangSmith.") 