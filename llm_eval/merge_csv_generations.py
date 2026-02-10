import pandas as pd


def merge_csvs(local_gen: str, euroeval_gen: str, output_path: str) -> pd.DataFrame:
    local_gen_df = pd.read_csv(local_gen)
    euroeval_gen_df = pd.read_csv(euroeval_gen)

    print(f"N rows CSV 1: {len(local_gen_df)}")
    print(f"N rows CSV 2: {len(euroeval_gen_df)}")

    if len(local_gen_df) != len(euroeval_gen_df):
        raise Exception("Row number mismatch")

    # merge adding columns from euroeval_gen_df to local_gen_df
    merged_df = pd.concat([local_gen_df, euroeval_gen_df], axis=1)

    # # merge dfs on columns "text_2" from local_gen_df and "text" from euroeval_gen_df
    # merged_df = pd.merge(local_gen_df, euroeval_gen_df, left_on="text_2", right_on="text", how="inner")

    # save merged df to csv with ; as separator
    merged_df.to_csv(output_path, index=False, sep=';')

    return merged_df


def main():
    local_generations = "logs/comma-v0.1-1t_log.csv"
    euroeval_generations = "logs/dala-gen-dataset_predictions_custom_0.csv"
    output_path = "logs/updated_csv.csv"

    merge_csvs(local_generations, euroeval_generations, output_path)

    # Process files
    try:
        print(f"\nSuccess! Updated CSV saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    main()
