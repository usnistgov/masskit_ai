import random
import timeit
import hydra
from hydra.utils import to_absolute_path
import logging
import pandas as pd
from omegaconf import DictConfig
import pyarrow.parquet as pq
from masskit.data_specs.schemas import min_spectrum_fields
from masskit.data_specs.spectral_library import LibraryAccessor
from masskit.utils.files import load_msp2array, load_sdf2array, load_mgf2array
from masskit.utils.hitlist import CosineScore
from masskit.utils.index import ArrowLibraryMap, PandasLibraryMap
from masskit.utils.general import class_for_name, parse_filename

"""
AIMS: search spectra using an index

to run two sequential searches with the same random query ids, first run with input.query.num set to a number of
random queries.  Save the results and then load them in using input.query.id_file set to the name of the output file
from the first search.
"""


@hydra.main(config_path="conf", config_name="config")
def aims_app(config: DictConfig) -> None:
    # Allow files relative to original execution directory
    search_file = to_absolute_path(config.input.search.file)
    query_file = to_absolute_path(config.input.query.file)
    index_file = to_absolute_path(config.input.index.file)

    # load the library as a LibraryMap
    search_map = ArrowLibraryMap(pq.read_table(search_file))

    # load query spectra, either as parquet file, pandas pickle, sdf, mgf or msp. Turn into a list of spectrum objects
    if query_file == search_file:
        query_map = search_map
    else:
        input_file_root, input_file_extension = parse_filename(query_file)
        if input_file_extension == 'msp':
            query_map = ArrowLibraryMap(load_msp2array(query_file, num=config.input.query.num))
        elif input_file_extension == 'sdf':
            query_map = ArrowLibraryMap(load_sdf2array(query_file, num=config.input.query.num))
        elif input_file_extension == 'pkl':
            df = pd.read_pickle(query_file)
            if config.input.query.num:
                df = df[0:config.input.query.num]
            query_map = PandasLibraryMap(df, column_name=config.input.query.column_name)
        elif input_file_extension == 'mgf':
            query_map = ArrowLibraryMap(load_mgf2array(query_file, num=config.input.query.num))
        elif input_file_extension == 'parquet':
            query_map = ArrowLibraryMap(pq.read_table(query_file))
        else:
            raise NotImplementedError(f'unable to read {query_file}')

    # load the index
    start_time = timeit.default_timer()
    index = class_for_name(config.paths.modules.indices, config.search.index_type.name)(config.input.index.name)
    index.load(file=index_file)
    elapsed = timeit.default_timer() - start_time
    logging.info(f"index loading time = {elapsed}")

    # obtain the query objects
    id_list_query = None
    query_num = config.input.query.num if config.input.query.num else len(query_map)
    if config.input.query.ids:
        queries = [query_map.getspectrum_by_id(x) for x in config.input.query.ids]
    elif config.input.query.id_file:
        # load in the queries from a search result file
        df_query = pd.read_pickle(to_absolute_path(config.input.query.id_file))
        query_ids = df_query.index.get_level_values(0).unique()
        queries = [query_map.getspectrum_by_id(x) for x in query_ids]
    elif config.input.query.column_name:
        # this is a bit of a hack to allow searching feature vectors, etc., instead of spectra.  
        # all of the other clauses in the if-then statement generate spectrum objects
        queries = query_map.to_arrow().slice(0, query_num)[config.input.query.column_name].to_numpy()
        # use a query id list to convert row number to query_id.  not needed in other clauses as 
        # spectrum objects contain the query id
        id_list_query = query_map.get_ids()
    else:
        # select a random set of query ids
        queries = [query_map[x] for x in random.sample(range(len(query_map)), query_num)]

    # noise filter the queries
    # queries = [query.norm(max_intensity_in=1.0).filter(min_mz=0.01).windowed_filter() for query in queries]

    # do the search
    if config.search.row_ids:
        id_list_query = None
        id_list = None
    else:
        id_list_query = query_map.get_ids()
        id_list = search_map.get_ids()
     
    start_time = timeit.default_timer()
    hitlist = index.search(queries, id_list=search_map.get_ids(), id_list_query=id_list_query, hitlist_size=config.search.hitlist.size)
    elapsed = timeit.default_timer() - start_time
    logging.info(f"search time per query = {elapsed / len(queries)}")

    start_time = timeit.default_timer()
    # score the hitlist if not already calculated
    if config.search.score.column_name == 'cosine_score':
        if "cosine_score" not in hitlist.to_pandas().columns:
            CosineScore(search_map, query_table_map=query_map, score_name=config.search.score.column_name).score(hitlist)
    elapsed = timeit.default_timer() - start_time
    logging.info(f"scoring time per query = {elapsed / len(queries)}")

    # sort the hitlist
    hitlist.sort(score=config.search.score.column_name)

    if query_file == search_file:
        logging.info(f"exact match count = "
                     f"{(hitlist.to_pandas().index.get_level_values(0) == hitlist.to_pandas().index.get_level_values(1)).sum()}")

    # write out hitlist as csv or pandas dataframe.
    if config.output.pkl.file:
        hitlist.save(config.output.pkl.file)
    if config.output.csv.file:
        hitlist.to_pandas().to_csv(config.output.csv.file)


if __name__ == "__main__":
    aims_app()
