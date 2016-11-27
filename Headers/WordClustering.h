/////#ifndef UTILS
/////#include "../Document_Data_Extraction_Visual/Utils.h"
/////#endif
/////#include "../Document_Data_Extraction_Visual/Params.h";
/////#include "../OntologyAPI/OntologyAPI.h"
#ifndef UTILS
#include "Utils.h"
#endif
#include "Params.h"
#include "OntologyAPI.h"
#include <map>
#include <vector>
#include <list>
#pragma once
#include "VisualWords.hpp"

// TODO: This class should be merged with the BoundingBox class.
/// Container with all the information extracted from a word snippet.
class WordInformation
{
public:
    // -[ Constructor, destructor and assignation operator ]-----------------------------------------------------------------------------------------
    /// Default constructor.
    WordInformation(void) : m_cluster_ids(-1) {}
    /// Constructor which initializes the word information object.
    WordInformation(const std::string &word_uris, const BoundingBox &coordinates) : m_word_uris(word_uris), m_coordinates(coordinates), m_cluster_ids(-1) {}
    
    // -[ Access functions ]-------------------------------------------------------------------------------------------------------------------------
    /// Returns the URI of the word.
    inline const std::string& getWordURI(void) const { return m_word_uris; }
    /// Sets the URI of the word.
    inline void setWordURI(const std::string &word_uris) { m_word_uris = word_uris; }
    /// Returns the bounding box of the word snippet.
    inline const BoundingBox& getBoundingBox(void) const { return m_coordinates; }
    /// Sets the bounding box information of the word snippet.
    inline void setBoundingBox(const BoundingBox &coordinates) { m_coordinates = coordinates; }
    /// Returns a constant reference to the BoVW signature of the word snippet.
    inline const std::vector<std::tuple<double, unsigned int> >& getSignature(void) const { return m_bovw_signature; }
    /// Sets the BoVW signature from a sparse-vector.
    inline void setSignature(const std::vector<std::tuple<double, unsigned int> > &bovw_signature) { m_bovw_signature = bovw_signature; }
    /// Sets the BoVW signature from a dense-vector.
    void setSignature(const std::vector<double> &bovw_signature);
    /// Returns the visual cluster identifier.
    inline unsigned int getVisualClusterID(void) const { return m_cluster_ids; }
    /// Sets the visual cluster identifier.
    inline void setVisualClusterID(unsigned int cluster_ids) { m_cluster_ids = cluster_ids; }
protected:
    /// URI of the word.
    std::string m_word_uris;
    /// Coordinates of the word in the image.
    BoundingBox m_coordinates;
    /// BoVW signature of the word.
    std::vector<std::tuple<double, unsigned int> > m_bovw_signature;
    /// Identifier of the cluster assigned to the word.
    unsigned int m_cluster_ids;
    // TODO: ADD CLUSTER REPRESENTATIVES
};

class CWordClustering
{
public:
    COntologyAPI * ontAPI;
    bool m_save_to_ontology;
    
    CWordClustering(void);
    ~CWordClustering(void);
    void ProcessCluster(int numcell);
    /// Sets the ontology object used to retrieve and modify the selected document database.
    inline void SetOntologyAPI(COntologyAPI& ontApi) { ontAPI = &ontApi; }
    /// Sets the parameters with the information of the collection to be processed.
    inline void SetConfigParams(CParams parameters) { m_cfg_params = parameters;}
    void LoadResults(void);
    void LoadClusters(int numcol);
    void SavetoOntology(void);
protected:
    template <class TDATA>
    TDATA getParameter(const char * parameter, TDATA default_value) const;
    template <class TDATA> TDATA convert(const std::string &value) const;
    
    //std::map<std::string, BoundingBox> m_result_boxes;
    //std::map<std::string, std::vector<std::string>> m_cluster_representatives;
    //std::vector<std::string> m_cluster_ids;
    //std::vector<std::string> m_word_uris;
    //std::map<std::string, std::string> cluster_transcription;
    std::list<WordInformation *> m_word_information;
    UPGMA m_upgma;
    CParams m_cfg_params;
    int col_to_be_processed;
};

template <class TDATA>
TDATA CWordClustering::getParameter(const char * parameter, TDATA default_value) const
{
    std::map<std::string, std::vector<std::string> >::const_iterator search;
    
    if ((search = m_cfg_params.cf_params.find(parameter)) == m_cfg_params.cf_params.end()) return default_value;
    else return convert<TDATA>(search->second[0]);
}

