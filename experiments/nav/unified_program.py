#!/usr/bin/env python3
"""Compile challenge language into the closed scene-graph query vocabulary."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from question_types import QuestionType, classify_question


RELATIONS = {"on", "above", "below", "under", "near", "between", "with_on"}
ANSWER_OPS = {"count", "argmin_dist", "argmax_dist", "unique", "path"}
SELECTOR_OPS = {"all", "unique", "argmin_dist", "argmax_dist"}
STRUCTURES = {"floor", "wall", "ceiling"}
LEG_KINDS = {"goto_near", "stop_at", "pass_between", "avoid_between",
             "pass_near", "avoid_near"}


def entity_dependency_closure(program: dict[str, Any], roots) -> list[str]:
    """All class entities needed to resolve ``roots`` faithfully.

    Relations and closest/farthest selectors turn a flat JSON list into a
    query graph.  Count, reference and follow must traverse the same graph;
    otherwise a follow leg such as "the stool under the picture" searches for
    stools but silently drops the picture and the ``under`` constraint.
    """
    entities = program.get("entities", {})
    dependencies = {str(root) for root in roots if root in entities}
    changed = True
    while changed:
        changed = False
        for predicate in program.get("filter", []):
            args = predicate.get("args", [])
            if args and args[0] in dependencies:
                for entity_id in args[1:]:
                    if entity_id in entities and entity_id not in dependencies:
                        dependencies.add(entity_id)
                        changed = True
        for entity_id, selector in program.get("selectors", {}).items():
            anchor = selector.get("to") if isinstance(selector, dict) else None
            if (entity_id in dependencies and anchor in entities and
                    anchor not in dependencies):
                dependencies.add(anchor)
                changed = True
    return sorted(entity_id for entity_id in dependencies
                  if isinstance(entities.get(entity_id, {}).get("class"), str))


COMPILE_SYSTEM = """You compile one indoor robotics question into a typed scene-
graph program. This is a text-only compiler: you do not see the room and must
never invent a quantity, color, object, relation, or location absent from the
question. Use only the supplied schema and closed operator vocabulary. Reply
with one JSON object and no prose."""


def compile_prompt(question: str, task: str, error: str = "") -> str:
    repair = f"\nPrevious validation error: {error}\nRepair it.\n" if error else ""
    return f"""QUESTION: {question}
TASK TYPE: {task}
{repair}
Return this schema:
{{
  "task": "count|refer|follow",
  "entities": {{
    "E1": {{"class":"singular head noun", "attributes":[],
             "sam_queries":["class", "useful synonym"]}},
    "F": {{"structure":"floor|wall|ceiling"}}
  }},
  "filter": [{{"op":"on|above|below|under|near|between|with_on",
                "args":["entity ids"]}}],
  "selectors": {{
    "optional entity id": {{"op":"all|unique|argmin_dist|argmax_dist",
                              "to":"entity id for distance selector"}}
  }},
  "answer": {{"op":"count|unique|argmin_dist|argmax_dist|path",
               "of":"E1", "to":"optional entity id", "legs":[]}}
}}

For closest use argmin_dist; farthest uses argmax_dist. Uninstanced support
references such as floor and wall MUST be entities with a structure field. A
structure is ONLY floor, wall, or ceiling. A bed, sofa, table, cabinet, shelf,
window, door, column, screen, or any other countable/groundable landmark is a
class entity, even when another object is described as being on it, because it
must be detected and grounded. Compile nested
references recursively. Every filter arg and answer.to MUST be an entity ID
declared in entities, never a class word. A closest/farthest phrase belongs in
the answer operator; do not also add a near filter for the same anchor.
For path, legs are ordered objects with kind among goto_near, stop_at,
pass_between, avoid_between, pass_near, avoid_near and an `of` entity id/list.
"take the path near X" and "pass by X" are pass_near. "avoid the path
near X" is avoid_near. Use pass_between/avoid_between only with two anchors.
Every noun phrase that participates in a relation is its own entity. Predicates
may constrain ANY class entity, not only answer.of. For example, "pillows on
the sofa under the pictures" requires on(pillow, sofa) AND under(sofa,
pictures). "Chairs with pillows on them" requires with_on(chair, pillow).
If closest/farthest selects a NESTED entity, put that comparison in selectors,
not in answer: "monitors on the table closest to the map decal" has answer
count(monitors), on(monitors, table), and selector table=argmin_dist(to decal).
Use answer argmin/argmax only when closest/farthest selects the final referred
object itself. Preserve every stated attribute and relation exactly once.
Do not put question answers or observed scene facts in the program."""


def _expected_task(question: str) -> str:
    kind = classify_question(question).question_type
    return {
        QuestionType.NUMERICAL: "count",
        QuestionType.OBJECT_REFERENCE: "refer",
        QuestionType.INSTRUCTION_FOLLOWING: "follow",
    }[kind]


def validate_program(value: dict[str, Any], question: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("program is not an object")
    task = value.get("task")
    expected = _expected_task(question)
    if task != expected:
        raise ValueError(f"task must be {expected}, got {task!r}")
    entities = value.get("entities")
    if not isinstance(entities, dict) or not entities:
        raise ValueError("entities must be a non-empty object")
    for entity_id, spec in entities.items():
        if not isinstance(spec, dict):
            raise ValueError(f"entity {entity_id} is not an object")
        has_class = isinstance(spec.get("class"), str) and bool(spec["class"].strip())
        has_structure = spec.get("structure") in STRUCTURES
        if has_class == has_structure:
            raise ValueError(
                f"entity {entity_id} needs exactly one of class or structure")
        if has_class:
            spec["class"] = spec["class"].strip().lower()
            attrs = spec.get("attributes", [])
            if not isinstance(attrs, list) or not all(isinstance(x, str) for x in attrs):
                raise ValueError(f"entity {entity_id} attributes must be strings")
            queries = spec.get("sam_queries") or [spec["class"]]
            if not isinstance(queries, list) or not all(
                    isinstance(x, str) and x.strip() for x in queries):
                raise ValueError(f"entity {entity_id} sam_queries invalid")
            spec["sam_queries"] = list(dict.fromkeys(x.strip().lower() for x in queries))
    filters = value.get("filter", [])
    if not isinstance(filters, list):
        raise ValueError("filter must be a list")
    for predicate in filters:
        if not isinstance(predicate, dict) or predicate.get("op") not in RELATIONS:
            raise ValueError(f"invalid filter {predicate!r}")
        args = predicate.get("args")
        if not isinstance(args, list) or not args or any(a not in entities for a in args):
            raise ValueError(f"filter references unknown entities: {predicate!r}")
        required_arity = 3 if predicate["op"] == "between" else 2
        if len(args) != required_arity:
            raise ValueError(
                f"{predicate['op']} requires {required_arity} entity arguments, "
                f"got {len(args)}")
        if not isinstance(entities[args[0]].get("class"), str):
            raise ValueError("the first filter argument must be the filtered class entity")
    selectors = value.get("selectors", {})
    if selectors is None:
        selectors = {}
    if not isinstance(selectors, dict):
        raise ValueError("selectors must be an object")
    for entity_id, selector in selectors.items():
        if entity_id not in entities or not isinstance(
                entities[entity_id].get("class"), str):
            raise ValueError(f"selector target {entity_id!r} is not a class entity")
        if not isinstance(selector, dict) or selector.get("op") not in SELECTOR_OPS:
            raise ValueError(f"invalid selector for {entity_id}: {selector!r}")
        if selector["op"] in {"argmin_dist", "argmax_dist"}:
            if selector.get("to") not in entities:
                raise ValueError(
                    f"selector for {entity_id} references unknown anchor")
    value["selectors"] = selectors
    answer = value.get("answer")
    if not isinstance(answer, dict) or answer.get("op") not in ANSWER_OPS:
        raise ValueError("invalid answer operator")
    if answer["op"] != "path" and answer.get("of") not in entities:
        raise ValueError("answer.of references an unknown entity")
    if answer.get("to") is not None and answer["to"] not in entities:
        raise ValueError("answer.to references an unknown entity")
    if task == "count" and answer["op"] != "count":
        raise ValueError("count task requires count answer operator")
    if task == "follow" and answer["op"] != "path":
        raise ValueError("follow task requires path answer operator")
    if answer["op"] == "path":
        legs = answer.get("legs")
        if not isinstance(legs, list) or not legs:
            raise ValueError("path answer requires non-empty ordered legs")
        for leg in legs:
            if not isinstance(leg, dict) or leg.get("kind") not in LEG_KINDS:
                raise ValueError(f"invalid path leg {leg!r}")
            references = leg.get("of")
            references = [references] if isinstance(references, str) else references
            if (not isinstance(references, list) or not references or
                    any(entity_id not in entities for entity_id in references)):
                raise ValueError(f"path leg references unknown entities: {leg!r}")
            expected = 2 if leg["kind"] in {"pass_between", "avoid_between"} else 1
            if len(references) != expected:
                raise ValueError(
                    f"{leg['kind']} requires {expected} entity reference(s)")
    # A refer task that compiled to `count` would publish an Int32 instead of
    # the scored Marker, losing the whole question on a formatting mistake.
    if task == "refer" and answer["op"] not in {"unique", "argmin_dist",
                                                "argmax_dist"}:
        raise ValueError(
            "refer task requires unique/argmin_dist/argmax_dist, "
            f"got {answer['op']!r}")
    lowered = question.lower()
    distance_ops = {answer.get("op")} | {
        selector.get("op") for selector in selectors.values()
        if isinstance(selector, dict)}
    if task == "refer" and re.search(r"\bclosest\b|\bnearest\b", lowered):
        if "argmin_dist" not in distance_ops:
            raise ValueError(
                "closest reference requires argmin_dist at the selected entity")
    if task == "refer" and re.search(r"\bfarthest\b|\bfurthest\b", lowered):
        if "argmax_dist" not in distance_ops:
            raise ValueError(
                "farthest reference requires argmax_dist at the selected entity")
    return value


def semantic_audit_prompt(question: str, program: dict) -> str:
    """Ask a text-only critic whether compilation lost task semantics."""
    return f"""QUESTION: {question}

COMPILED PROGRAM:
{json.dumps(program, ensure_ascii=False)}

Audit semantic faithfulness, not JSON syntax. Check that every physical noun
phrase has the correct singular head class and attributes; every stated
on/above/below/under/near/between/with relation is represented with the right
direction; nested relations constrain the correct entity; closest/farthest is
attached to the entity it actually selects; and follow legs preserve order and
distinguish destinations from pass-by/path-near and avoid-near/between
constraints. Do not demand facts absent from the question.

Reply with JSON only:
{{"faithful": true|false, "missing_or_wrong": ["short exact issue"]}}"""


def repair_literal_entity_references(value: Any) -> Any:
    """Repair local-model schema slips without changing question semantics.

    The model often understands the semantics perfectly but writes a class
    literal where the schema requires an entity ID. Promote such literals to
    declared reference entities without inventing any scene fact. It also
    sometimes writes a physical support (for example a bed) in the closed
    ``structure`` field; preserve that literal as an ordinary class entity.
    """
    if not isinstance(value, dict) or not isinstance(value.get("entities"), dict):
        return value
    out = deepcopy(value)
    entities = out["entities"]
    answer = out.get("answer") if isinstance(out.get("answer"), dict) else {}

    for spec in entities.values():
        if not isinstance(spec, dict):
            continue
        structure = spec.get("structure")
        has_class = isinstance(spec.get("class"), str) and bool(
            spec["class"].strip())
        if (isinstance(structure, str) and structure.strip().lower()
                not in STRUCTURES and not has_class):
            literal = structure.strip().lower()
            spec.pop("structure", None)
            spec["class"] = literal
            spec.setdefault("attributes", [])
            spec.setdefault("sam_queries", [literal])

    # `to` has meaning only for a distance answer. Qwen often copies it from
    # the schema example into count/unique answers; deleting this dead field
    # cannot alter the executable query.
    if answer.get("op") not in {"argmin_dist", "argmax_dist"}:
        answer.pop("to", None)
    literal_ids: dict[str, str] = {}

    def promote(reference: Any) -> Any:
        if not isinstance(reference, str) or reference in entities:
            return reference
        literal = re.sub(r"^(?:the|a|an)\s+", "", reference.strip().lower())
        if not literal:
            return reference
        if literal in literal_ids:
            return literal_ids[literal]
        index = 1
        while f"R{index}" in entities:
            index += 1
        entity_id = f"R{index}"
        entities[entity_id] = {
            "class": literal, "attributes": [], "sam_queries": [literal]}
        literal_ids[literal] = entity_id
        return entity_id

    raw_to = answer.get("to")
    if raw_to is not None:
        answer["to"] = promote(raw_to)
    repaired_filters = []
    for predicate in out.get("filter", []) if isinstance(
            out.get("filter", []), list) else []:
        if not isinstance(predicate, dict):
            repaired_filters.append(predicate)
            continue
        op = predicate.get("op")
        raw_args = predicate.get("args")
        if not isinstance(raw_args, list):
            repaired_filters.append(predicate)
            continue
        # "closest to X" is argmin, not an additional 2 m near predicate.
        if (op == "near" and len(raw_args) == 1 and raw_to is not None and
                raw_args[0] == raw_to and
                answer.get("op") in {"argmin_dist", "argmax_dist"}):
            continue
        args = [promote(item) for item in raw_args]
        if len(args) == 1 and answer.get("of") in entities:
            args.insert(0, answer["of"])
        fixed = dict(predicate)
        fixed["args"] = args
        repaired_filters.append(fixed)
    out["filter"] = repaired_filters
    return out


def fallback_program(question: str) -> dict[str, Any]:
    """Legacy diagnostic parser retained for narrow unit tests only.

    Production deliberately does not use this: schema-valid regex output can
    erase nested relations and make the robot confidently solve another task.
    """
    task = _expected_task(question)
    text = " ".join(question.lower().split())
    if task == "count":
        body = re.sub(r"^(how many|count(?: the)?|what is the number of)\s+", "", text)
        relation = None
        structure = None
        for token, normalized in ((" on the floor", "floor"),
                                  (" on the wall", "wall"),
                                  (" under the", None), (" above the", None),
                                  (" on the", None), (" near the", None)):
            if token in body:
                body = body.split(token, 1)[0]
                if normalized:
                    relation, structure = "on", normalized
                break
        noun = re.sub(r"\b(are|is|there|present|in this room|in the room)\b", "", body)
        noun = noun.strip(" ?.!")
        if noun.endswith("s") and not noun.endswith("ss"):
            noun = noun[:-1]
        entities: dict[str, Any] = {
            "E1": {"class": noun or "object", "attributes": [],
                   "sam_queries": [noun or "object"]}}
        filters = []
        if structure:
            entities["F"] = {"structure": structure}
            filters.append({"op": relation, "args": ["E1", "F"]})
        return {"task": "count", "entities": entities, "filter": filters,
                "answer": {"op": "count", "of": "E1"}}
    if task == "refer":
        relation = re.search(
            r"^(?:find\s+|the\s+)?(.+?)\s+(?:that\s+is\s+)?"
            r"(closest|nearest|farthest|furthest)\s+(?:to|from)\s+(.+?)[.?!]*$",
            text)
        if relation:
            target_phrase, direction, anchor_phrase = relation.groups()
            colors = {"red", "blue", "green", "white", "black", "gray",
                      "grey", "yellow", "orange", "brown", "pink", "purple"}
            target_words = re.findall(r"[a-z]+", target_phrase)
            attrs = [word for word in target_words if word in colors]
            target_class = " ".join(word for word in target_words
                                    if word not in colors) or "object"
            anchor = re.sub(r"^(?:the|a|an)\s+", "", anchor_phrase).strip(" .?!")
            entities = {
                "E1": {"class": target_class, "attributes": attrs,
                       "sam_queries": [target_class,
                                       " ".join(attrs + [target_class])]},
                "R1": {"class": anchor, "attributes": [],
                       "sam_queries": [anchor]},
            }
            op = ("argmin_dist" if direction in {"closest", "nearest"}
                  else "argmax_dist")
            return {"task": "refer", "entities": entities, "filter": [],
                    "answer": {"op": op, "of": "E1", "to": "R1"}}
        words = re.findall(r"[a-z]+", text)
        noun = words[-1] if words else "object"
        entities = {"E1": {"class": noun, "attributes": [],
                            "sam_queries": [noun]}}
        return {"task": "refer", "entities": entities, "filter": [],
                "answer": {"op": "unique", "of": "E1"}}
    words = re.findall(r"[a-z]+", text)
    noun = words[-1] if words else "object"
    entities = {"E1": {"class": noun, "attributes": [],
                        "sam_queries": [noun]}}
    return {"task": "follow", "entities": entities, "filter": [],
            "answer": {"op": "path", "legs": []}}


def compile_question(vlm, question: str) -> tuple[dict[str, Any], list[str]]:
    from agent import _json
    traces: list[str] = []
    error = ""
    last_valid = None
    for attempt in range(3):
        prompt = compile_prompt(question, _expected_task(question), error)
        raw = vlm._gen([
            {"role": "system", "content": [{"type": "text", "text": COMPILE_SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ], [], max_new_tokens=750, label="unified_compile",
            tag=f"attempt_{attempt + 1}")
        traces.append(raw)
        try:
            repaired = repair_literal_entity_references(_json(raw))
            program = validate_program(repaired, question)
            last_valid = program
            audit_raw = vlm._gen([
                {"role": "system", "content": [{"type": "text", "text":
                    "You audit semantic parses of robotics language. Return JSON only."}]},
                {"role": "user", "content": [{"type": "text", "text":
                    semantic_audit_prompt(question, program)}]},
            ], [], max_new_tokens=350, label="unified_compile_audit",
                tag=f"attempt_{attempt + 1}")
            traces.append(audit_raw)
            audit = _json(audit_raw)
            if isinstance(audit, dict) and audit.get("faithful") is True:
                return program, traces
            issues = audit.get("missing_or_wrong", []) if isinstance(
                audit, dict) else ["semantic audit did not return valid JSON"]
            error = "semantic audit: " + "; ".join(map(str, issues))
        except (TypeError, ValueError) as exc:
            error = str(exc)
    # A schema-valid but semantically wrong fallback is more dangerous than a
    # visible compile failure: it makes the robot confidently solve a different
    # question. Keep the final candidate in the trace for diagnosis, but fail
    # closed instead of invoking the regex fallback in production.
    raise ValueError(
        f"could not produce a semantically faithful program after 3 attempts; "
        f"last_error={error}; last_valid={last_valid}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    print(json.dumps(fallback_program(args.question), indent=2))
