/**
 * @file embedding_writer.h
 * @brief contains an abstract logic class EmbeddingWriter and its child classes
 * @details child classes specify the format of file, i.e., the way that embeddings are stored
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#ifndef EMBEDDING_WRITER_H
#define EMBEDDING_WRITER_H

class EmbeddingWriter {
public:
    virtual void write(string file_path, Vertex* embeddings) = 0;
}

class PlainEmbeddingWriter : public EmbeddingWriter {
public:
    void write(string file_path, Vertex* embeddings) override;
}

#endif