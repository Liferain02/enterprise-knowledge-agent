"""Analysis of user-supplied CSV text; no file or database access."""
import csv
import io
import statistics

from src.agent.skills._validation import bounded_text, finite, json_result


def _table(csv_text: str):
    try:
        rows = list(csv.reader(io.StringIO(bounded_text(csv_text)), strict=True))
    except csv.Error as exc:
        raise ValueError(f"CSV 格式错误: {exc}") from exc
    if len(rows) < 2 or len(rows) > 1001:
        raise ValueError("CSV 必须包含表头和 1–1000 行数据")
    header = [name.strip() for name in rows[0]]
    if not all(header) or len(header) > 50 or len(set(header)) != len(header):
        raise ValueError("表头必须非空、唯一，且不超过 50 列")
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError("每行列数必须与表头一致")
    return header, [[cell.strip() for cell in row] for row in rows[1:]]


@json_result
def profile_csv(csv_text: str) -> str:
    """分析用户粘贴的 CSV 文本，返回行数、重复行、缺失值与数值列统计；不读取路径。"""
    header, rows = _table(csv_text)
    columns = {}
    for index, name in enumerate(header):
        values = [row[index] for row in rows if row[index] != ""]
        summary = {"missing": len(rows) - len(values), "nonempty": len(values)}
        try:
            numbers = [finite(value) for value in values]
        except ValueError:
            summary["type"] = "text_or_invalid_number"
        else:
            summary["type"] = "numeric" if numbers else "empty"
            if numbers:
                summary.update(min=min(numbers), max=max(numbers), mean=statistics.fmean(numbers))
        columns[name] = summary
    return {"rows": len(rows), "duplicate_rows": len(rows) - len(set(map(tuple, rows))), "columns": columns}


@json_result
def aggregate_csv(csv_text: str, group_column: str, value_column: str, operation: str = "mean") -> str:
    """按指定 CSV 列分组汇总数值列；operation 支持 mean/sum/min/max/count，空值跳过并报告。"""
    header, rows = _table(csv_text)
    operations = {"mean": statistics.fmean, "sum": sum, "min": min, "max": max, "count": len}
    if operation not in operations or group_column not in header or value_column not in header:
        raise ValueError("请提供存在的列名及 mean/sum/min/max/count 操作")
    gi, vi = header.index(group_column), header.index(value_column)
    groups = {}
    skipped = 0
    for row in rows:
        if not row[vi]:
            skipped += 1
            continue
        values = groups.setdefault(row[gi], [])
        if len(groups) > 50:
            raise ValueError("分组不能超过 50 个")
        values.append(finite(row[vi]))
    return {"operation": operation, "skipped_empty_values": skipped, "groups": [
        {"group": group, "count": len(values), "value": operations[operation](values)}
        for group, values in groups.items()
    ]}
