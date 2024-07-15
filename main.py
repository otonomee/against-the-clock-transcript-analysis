import os
from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


def preprocess_text(text):
    # Placeholder for actual preprocessing logic
    return text


def chunk_text(text, chunk_size=500):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(documents, embeddings)


def create_analysis_chain():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    template = """
    Analyze this 'Against the Clock' transcript excerpt:

    {text}

    Briefly identify the top 5 most interesting aspects of the artist's process in:
    1. Time management
    2. Sound design
    3. Rhythm construction
    4. Workflow optimization
    5. Creative problem-solving

    Be concise and specific.
    """
    prompt = ChatPromptTemplate.from_template(template)
    return LLMChain(llm=llm, prompt=prompt)


def analyze_documents(documents, analysis_chain, batch_size=5):
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_results = []
        for doc in batch:
            analysis = analysis_chain.run(text=doc.page_content)
            batch_results.append(
                {
                    "content": doc.page_content,
                    "analysis": analysis,
                    "metadata": doc.metadata,
                }
            )
        results.extend(batch_results)
    return results


def group_results_by_artist(results):
    grouped = {}
    for result in results:
        artist = os.path.basename(result["metadata"]["source"]).split(" - ")[0]
        if artist not in grouped:
            grouped[artist] = []
        grouped[artist].append(result)
    return grouped


def create_comparison_chain():
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    template = """
    Compare and contrast the following analyses from different 'Against the Clock' episodes:

    {analyses}

    Identify the top 5 most common and interesting elements across all episodes, focusing on:
    1. Time management techniques
    2. Sound design approaches
    3. Rhythmic construction methods
    4. Workflow optimization strategies
    5. Creative problem-solving tactics

    For each element, provide:
    - A brief description of the technique or approach
    - How frequently it appears across episodes
    - Why it's significant or effective in the context of rapid music production

    Present your findings in a clear, concise manner.
    """
    prompt = ChatPromptTemplate.from_template(template)
    return LLMChain(llm=llm, prompt=prompt)


def generate_comparative_insights(grouped_results, comparison_chain):
    comparative_insights = []
    for artist, artist_results in grouped_results.items():
        analyses = "\n\n".join([r["analysis"] for r in artist_results])
        insight = comparison_chain.run(analyses=analyses)
        comparative_insights.append({"artist": artist, "insight": insight})
    return comparative_insights


def create_report_chain():
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    template = """
    Based on the comparative insights from multiple 'Against the Clock' episodes, provide a comprehensive report on the top 5 most significant common elements in the artists' processes.

    {insights}

    For each of the top 5 elements:
    1. Provide a clear, concise description of the technique or approach
    2. Explain why it's particularly effective or innovative in rapid music production
    3. Give specific examples of how different artists have used or adapted this element
    4. Discuss any variations or evolutions of this element across different episodes or genres

    Conclude with a summary of how these top 5 elements contribute to successful rapid music production in the 'Against the Clock' series.

    Present your report in a clear, structured manner with headings for each of the top 5 elements.
    """
    prompt = ChatPromptTemplate.from_template(template)
    return LLMChain(llm=llm, prompt=prompt)


def generate_final_report(comparative_insights, report_chain):
    all_insights = "\n\n".join(
        [f"Artist: {ci['artist']}\n{ci['insight']}" for ci in comparative_insights]
    )
    return report_chain.run(insights=all_insights)


def main():
    directory = "preprocessed_transcripts"
    all_results = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                content = file.read()
            processed_doc = preprocess_text(content)
            chunks = chunk_text(processed_doc)
            documents = [
                Document(page_content=chunk, metadata={"source": filename})
                for chunk in chunks
            ]

            vectorstore = create_vectorstore(documents)
            analysis_chain = create_analysis_chain()
            results = analyze_documents(documents, analysis_chain)

            all_results.extend(results)

    # Now that we have results from all files, we can perform aggregate analysis
    grouped_results = group_results_by_artist(all_results)
    comparison_chain = create_comparison_chain()
    comparative_insights = generate_comparative_insights(
        grouped_results, comparison_chain
    )

    report_chain = create_report_chain()
    final_report = generate_final_report(comparative_insights, report_chain)

    # Save results
    with open("analysis_results/comparative_insights.txt", "w") as f:
        for insight in comparative_insights:
            f.write(f"Artist: {insight['artist']}\n")
            f.write(insight["insight"])
            f.write("\n\n")

    with open("analysis_results/final_report.txt", "w") as f:
        f.write(final_report)

    print(
        "Analysis complete. Results saved to 'comparative_insights.txt' and 'final_report.txt'."
    )


if __name__ == "__main__":
    main()
