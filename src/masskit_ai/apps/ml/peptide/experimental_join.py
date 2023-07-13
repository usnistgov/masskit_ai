#!/bin/env python
import os
import pyarrow as pa
import pyarrow.parquet as pq

if __name__ == "__main__":
    DATA_DIR = r'/aiomics/massspec_cache/nist/aiomics'
    FILES = [ "AItrain_All.parquet", "TestUniq202277.parquet", "ValidUniq2022418.parquet" ]
    OUTNAME = "dataset.parquet"

    def check_columns(d):
        names = {}
        for k,v in d.items():
            names[k] = set(v.schema.to_arrow_schema().names)
        common = names[FILES[0]] & names[FILES[1]] & names[FILES[2]]
        #uncommon = (names[FILES[0]] ^ names[FILES[1]] ) & ( names[FILES[1]] ^ names[FILES[2]] )

        # These fields caused problems, but they are not needed for this issue so we ignore them
        common.remove("set")
        common.remove("collision_energy")
        common.remove("annotations") 

        return list(common)

    def get_metadata():
        md_dict = {}
        for file in FILES:
            table_name = os.path.join(DATA_DIR, file)
            metadata = pq.read_metadata(table_name)
            md_dict[file] = metadata
        #print(file, metadata.schema.to_arrow_schema().names)
        return md_dict

    def read_data(columns):
        tables = []
        for file in FILES:
            table_name = os.path.join(DATA_DIR, file)
            subtable = pq.read_table(table_name, columns=columns)
            #print(subtable.column("set"))
            tables.append(subtable)
        table = pa.concat_tables(tables, promote=True)

        print(f"Read {table.num_rows} rows from {len(FILES)} files")
        return table


    md = get_metadata()
    columns = check_columns(md)

    table = read_data(columns)
    mods = table.column("mod_names")

    vals = []
    for mod in mods:
        vals.append(21 in mod.as_py())
    is_phospho = pa.array(vals)

    table = table.append_column("is_phospho", is_phospho)

    pq.write_table(table, OUTNAME)
    print(f"Wrote a table with {table.num_columns} cols and {table.num_rows} rows to {OUTNAME}")
