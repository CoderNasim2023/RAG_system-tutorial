import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone'

//Step 1: Load the PDF file
async function IndexDocument() {
    
    const PDF_PATH = './ideapad.pdf';
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();
    console.log("PDF successfully loaded")
  
    //Now we've to do chunking the pdf 
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 700,
        chunkOverlap: 400,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log("chunking completed")

    //Step 3: Initializing the Embedding model

    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });

    console.log("Embeddidng model successfully configured")

    //Step4:  Initialize Pinecone Client
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    console.log("Pinecone model configured")

    ///langchain (chunking, embedding,  vector database)
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
        pineconeIndex,
        maxConcurrency: 5,
    });

    console.log("data stored successfully")
}

IndexDocument();