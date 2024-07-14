import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()

# Ensure OPENAI_API_KEY is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def load_and_split_documents(directory):
    loader = DirectoryLoader(directory, glob="*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(splits, embeddings)

def create_analysis_chain():
    llm = ChatOpenAI(temperature=0)
    template = """
    Analyze the following transcript excerpt from the 'Against the Clock' series:

    {text}

    Please provide:
    1. Key equipment mentioned
    2. Main techniques or processes described
    3. Artist's approach or philosophy
    4. Any unique or innovative methods
    5. Challenges faced during the process
    6. Overall mood or sentiment

    Be concise but insightful in your analysis.
    """
    prompt = ChatPromptTemplate.from_template(template)
    return LLMChain(llm=llm, prompt=prompt)

def analyze_documents(splits, analysis_chain):
    results = []
    for doc in splits:
        analysis = analysis_chain.run(text=doc.page_content)
        results.append({
            "content": doc.page_content,
            "analysis": analysis,
            "metadata": doc.metadata
        })
    return results

def group_results_by_artist(results):
    grouped = {}
    for result in results:
        artist = os.path.basename(result['metadata']['source']).split(' - ')[0]
        if artist not in grouped:
            grouped[artist] = []
        grouped[artist].append(result)
    return grouped

def create_comparison_chain():
    llm = ChatOpenAI(temperature=0)
    template = """
    Compare and contrast the following analyses from different 'Against the Clock' episodes:

    {analyses}

    Provide insights on:
    1. Common themes or approaches
    2. Unique or standout methods
    3. Evolution of techniques (if apparent)
    4. Diversity of equipment choices
    5. Overall trends in creative processes

    Summarize your findings in a concise but informative manner.
    """
    prompt = ChatPromptTemplate.from_template(template)
    return LLMChain(llm=llm, prompt=prompt)

def generate_comparative_insights(grouped_results, comparison_chain):
    comparative_insights = []
    for artist, artist_results in grouped_results.items():
        analyses = "\n\n".join([r['analysis'] for r in artist_results])
        insight = comparison_chain.run(analyses=analyses)
        comparative_insights.append({
            "artist": artist,
            "insight": insight
        })
    return comparative_insights

def create_report_chain():
    llm = ChatOpenAI(temperature=0)
    template = """
    Based on the comparative insights from multiple 'Against the Clock' episodes, provide a comprehensive report on:

    {insights}

    The report should cover:
    1. Overall trends in music production techniques
    2. Common challenges and solutions
    3. Innovative approaches observed
    4. Equipment preferences and their impact
    5. The role of time constraints in the creative process
    6. Diversity of styles and methods across artists
    7. Any other significant observations

    Organize the report in a clear, structured manner with headings and subheadings.
    """
    prompt = ChatPromptTemplate.from_template(template)
    return LLMChain(llm=llm, prompt=prompt)

def generate_final_report(comparative_insights, report_chain):
    all_insights = "\n\n".join([f"Artist: {ci['artist']}\n{ci['insight']}" for ci in comparative_insights])
    return report_chain.run(insights=all_insights)

def main():
    # Load and split documents
    splits = load_and_split_documents('transcripts')
    
    # Create vectorstore
    vectorstore = create_vectorstore(splits)
    
    # Analyze documents
    analysis_chain = create_analysis_chain()
    results = analyze_documents(splits, analysis_chain)
    
    # Group results by artist
    grouped_results = group_results_by_artist(results)
    
    # Generate comparative insights
    comparison_chain = create_comparison_chain()
    comparative_insights = generate_comparative_insights(grouped_results, comparison_chain)
    
    # Generate final report
    report_chain = create_report_chain()
    final_report = generate_final_report(comparative_insights, report_chain)
    
    # Save results
    with open('comparative_insights.txt', 'w') as f:
        for insight in comparative_insights:
            f.write(f"Artist: {insight['artist']}\n")
            f.write(insight['insight'])
            f.write('\n\n')
    
    with open('final_report.txt', 'w') as f:
        f.write(final_report)
    
    print("Analysis complete. Results saved to 'comparative_insights.txt' and 'final_report.txt'.")

if __name__ == "__main__":
    main()
