from context.passFinderContext import extract_context_passfinder_all, merge_passfinder_context

if __name__ == '__main__':
    language = 'cpp'
    base = f'/home/gonghuihui/pwd_proj/{language}_database'
    extract_context_passfinder_all(
        base,
        language,
        "pass"
    )

    extract_context_passfinder_all(
        base,
        language,
        "string"
    )

