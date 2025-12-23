class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, question):
        ctx = self.retriever.retrieve(question)
        ans = self.generator.generate(ctx, question)
        return ctx, ans
